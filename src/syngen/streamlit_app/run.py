import os
import time
from threading import Thread

import streamlit as st
from streamlit_option_menu import option_menu

from syngen.streamlit_app.handlers import StreamlitHandler
from syngen.streamlit_app.utils import (
    show_data,
    get_running_status,
    set_session_state,
    cleanup_artifacts
)
from syngen import __version__


class StreamlitApp:

    @staticmethod
    def setup_ui():
        """
        Set up the UI for the Streamlit app
        """
        path_to_logo = f"{os.path.join(os.path.dirname(__file__))}/img/logo.svg"
        path_to_logo_img = f"{os.path.join(os.path.dirname(__file__))}/img/favicon.svg"
        st.set_page_config(
            page_title="SynGen UI",
            page_icon=path_to_logo_img
        )
        st.sidebar.image(path_to_logo, use_column_width=True)
        st.markdown(
            f"""
                <style>
                {"".join(open(f"{os.path.join(os.path.dirname(__file__))}/css/style.css").readlines())}
                </style>
                """, unsafe_allow_html=True
        )

        st.title(
            "SynGen UI",
            help=f"The current version of syngen library is {__version__}"
        )

    @staticmethod
    def handle_cross_icon():
        """
        Handle the behavior of disabling of the cross icon
        """
        running_status = get_running_status()

        display_status = "none" if running_status else "block"

        css = f"""
            <style>
                div[data-testid="fileDeleteBtn"] {{
                display: {display_status};
                }}
            </style>
            """
        st.markdown(css, unsafe_allow_html=True)

    @staticmethod
    def _get_streamlit_handler(epochs, size_limit, reports, uploaded_file):
        """
        Get the Streamlit handler
        """
        return StreamlitHandler(epochs, size_limit, reports, uploaded_file)

    def run_basic_page(self):
        """
        Run the basic page of the Streamlit app
        """
        set_session_state()
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type="csv",
            accept_multiple_files=False,
            disabled=get_running_status(),
        )
        self.handle_cross_icon()
        if not uploaded_file:
            cleanup_artifacts()
            st.warning("Please upload a CSV file to proceed")
        if uploaded_file:
            show_data(uploaded_file)
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                value=1,
                help="- The larger number of epochs is set the better training result is.\n"
                     "- The larger number of epochs is set the longer time for training "
                     "will be required.\n"
                     "- Actual number of epochs can be smaller that the one that was set here. "
                     "Once training stops improving the model, further training is not needed.",
                disabled=get_running_status()
            )
            size_limit = st.number_input(
                "Rows to generate",
                min_value=1,
                max_value=None,
                value=1000,
                disabled=get_running_status()
            )
            reports = st.checkbox(
                "Create an accuracy report",
                value=False,
                key="reports",
                disabled=get_running_status()
            )
            handler = self._get_streamlit_handler(epochs, size_limit, reports, uploaded_file)
            if st.button(
                    "Generate data",
                    type="primary",
                    key="gen_button",
                    disabled=get_running_status()
            ):
                runner = Thread(name="train_and_infer", target=handler.train_and_infer)
                runner.start()
                current_progress = 0
                prg = st.progress(current_progress)

                while runner.is_alive():
                    with st.expander("Logs"):
                        while True:
                            if not handler.log_queue.empty():
                                with st.code("logs", language="log"):
                                    log = handler.log_queue.get()
                                    st.text(log)
                                    current_progress, message = handler.progress_handler.info
                                    prg.progress(value=current_progress, text=message)
                            elif not handler.log_error_queue.empty():
                                runner.join()
                                break
                            elif not runner.is_alive():
                                break
                            time.sleep(0.001)
                if not handler.log_error_queue.empty() and not runner.is_alive():
                    st.exception(handler.log_error_queue.get())
                elif handler.log_queue.empty() and not runner.is_alive():
                    prg.progress(100)
                    st.success("Data generation completed")

            with st.container():
                handler.generate_buttons()

    def run(self):
        """
        Run the Streamlit app
        """
        self.setup_ui()
        with st.sidebar:
            selected = option_menu("", ["Basic"],
                                   icons=["'play'"],
                                   default_index=0,
                                   menu_icon=None,
                                   styles={
                                       "container": {"font-family": "Open Sans"}
                                   }
                                   )
        if selected == "Basic":
            self.run_basic_page()


if __name__ == "__main__":
    StreamlitApp().run()
