let scrollTO = (id, scrollToId) =>
	document.getElementById(id).addEventListener("click", () => {
		document.getElementById(scrollToId).scrollIntoView({ behavior: "smooth" });
	});

scrollTO("Correlations", "h2-title--correlations");
scrollTO("Clustering", "h2-title-clustering");
scrollTO("Utility", "h2-title-utility");
scrollTO("Univariate", "h2-title-univariate");
scrollTO("Bivariate", "h2-title-bivariate");

