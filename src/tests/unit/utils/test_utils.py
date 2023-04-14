from unittest import TestCase
from unittest.mock import Mock
from syngen.ml.utils import slugify_attribute, slugify_parameters


class TestSlugify(TestCase):
    def test_slugify_attribute(self):
        mock = Mock(
            attr_1="My Test Attribute",
            attr_2="Мой другой аттрибут",
            attr_3="@#$12345*&^"
        )

        @slugify_attribute(attr_1="slug_attr1", attr_2="slug_attr2", attr_3="slug_attr3")
        def dummy_function(mock):
            pass

        dummy_function(mock)

        assert mock.slug_attr1, "my-test-attribute"
        assert mock.slug_attr2, "moi-drugoi-attribut"
        assert mock.slug_attr3, "12345"

    def test_slugify_parameters(self):
        @slugify_parameters("name")
        def dummy_function(name):
            return name

        assert dummy_function(name="My Test Attribute"), "my-test-attribute"
        assert dummy_function(name="Мой другой аттрибут"), "moi-drugoi-attribut"
        assert dummy_function(name="@#$12345*&^"), "12345"
