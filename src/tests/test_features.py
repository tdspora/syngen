from unittest import TestCase
import pandas as pd


from syngen.ml.vae.models.features import DateFeature


class TestDateFeature(TestCase):
    def test_validate_date_format_1(self):
        data = pd.DataFrame({'Date':['01-01-2020', '02/02/2000', '05-05-2020']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%m-%d-%Y"

    def test_validate_date_format_2(self):
        data = pd.DataFrame({'Date':['31-01-2020', '20/02/2000', '25-05-2020']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%d-%m-%Y"

    def test_validate_date_format_3(self):
        data = pd.DataFrame({'Date':['03/03/2000', '01/01/2020', '05-05-2020']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%m/%d/%Y"

    def test_validate_date_format_4(self):
        data = pd.DataFrame({'Date':['31/01/2020', '20/02/2000', '25/05/2020']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%d/%m/%Y"

    def test_validate_date_format_5(self):
        data = pd.DataFrame({'Date':['2020/01/01', '1999/01/09', '05-05-2020']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%Y/%m/%d"

    def test_validate_date_format_6(self):
        data = pd.DataFrame({'Date':['2020-01-01', '1999-01-09', '05-05-2020']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%Y-%m-%d"

    def test_validate_date_format_7(self):
        data = pd.DataFrame({'Date':['March 10, 2022', 'September 11, 1900', 'May 15, 1877']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%B %d, %Y"

    def test_validate_date_format_8(self):
        data = pd.DataFrame({'Date':['March 10, 2022', 'September 11, 1900', 'May 15, 1877']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%B %d, %Y"

    def test_validate_date_format_9(self):
        data = pd.DataFrame({'Date': ['Jul 10, 2022', 'Jan 11, 1900', 'Feb 15, 1877']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%b %d, %Y"

    def test_validate_date_format_10(self):
        data = pd.DataFrame({'Date': ['10 June 2022', '11 January 1900', '01 February 1877']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%d %B %Y"

    def test_validate_date_format_11(self):
        data = pd.DataFrame({'Date': ['Jul 10 2022', 'Jan 11 1900', 'Feb 15 1877']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%b %d %Y"

    def test_validate_date_format_12(self):
        data = pd.DataFrame(
            {'Date':['1989-01-01 00:00:00.000000', '1897-01-01 03:03:00.000000', '2020-01-01 03:03:03.000000']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%Y-%m-%d"

    def test_validate_date_format_13(self):
        data = pd.DataFrame({'Date':['1989/01/01 00:00:00.000000', '1897/01/01 03:03:00.000000', '2020/01/01 03:03:03.000000']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%Y/%m/%d"

    def test_validate_date_format_14(self):
        data = pd.DataFrame(
            {'Date': ['2010-10-23 18:25:00 BRST', '2012-01-19 17:21:00 BRST', '2002-05-09 11:31:00 BRST']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%Y-%m-%d"

    def test_validate_date_format_15(self):
        data = pd.DataFrame(
            {'Date': ['2010-10-23 18:25:00 +0300', '2012-01-19 17:21:00 +0300', '2002-05-09 11:31:00 +0300']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%Y-%m-%d"

    def test_validate_date_format_16(self):
        data = pd.DataFrame(
            {'Date': ['2012/01/19 17:21:00', '2012/01/19 17:21:00', '2012/01/19 17:21:00']})
        date_feature = DateFeature(name="date_feature")
        date_feature.fit(data)
        assert date_feature.date_format == "%Y/%m/%d"
