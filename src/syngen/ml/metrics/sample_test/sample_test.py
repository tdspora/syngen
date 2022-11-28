import jinja2
import pandas as pd
import os
from syngen.ml.metrics import UnivariateMetric, BaseTest

from syngen.ml.metrics.utils import transform_to_base64


class SampleAccuracyTest(BaseTest):
    def __init__(self, original: pd.DataFrame, sampled: pd.DataFrame, paths: dict):
        super().__init__(original, sampled, paths)

    def __get_univariate_metric(self):
        """
        Do preparation work before creating the report
        """
        draws_path = self.paths["draws_path"]
        os.makedirs(draws_path, exist_ok=True)
        sample_acc_draws_path = f"{draws_path}/sample_accuracy"
        os.makedirs(sample_acc_draws_path, exist_ok=True)
        univariate = UnivariateMetric(self.original, self.synthetic, True, sample_acc_draws_path)
        return univariate


    def report(self, **kwargs):
        univariate = self.__get_univariate_metric()
        univariate.calculate_all(kwargs["cont_columns"], kwargs["categ_columns"])

        # Generate html report
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/sample_report_template.html") as file_:
            template = jinja2.Template(file_.read())

        draws_acc_path = f"{self.paths['draws_path']}/sample_accuracy"
        uni_images = [transform_to_base64(f"{draws_acc_path}/{f}") for f in os.listdir(draws_acc_path)
                      if f.startswith("univariate")]

        html = template.render(uni_imgs=uni_images)

        with open(f"{self.paths['draws_path']}/sample_accuracy_report.html", 'w') as f:
            f.write(html)
