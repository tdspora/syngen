import pytest
import numpy as np

from syngen.ml.flatten_json import (
    check_if_numbers_are_consecutive,
    flatten,
    flatten_preserve_lists,
    unflatten,
    unflatten_list,
    safe_flatten
)
from tests.conftest import SUCCESSFUL_MESSAGE


@pytest.mark.parametrize(
    "input, expected_result",
    [
        ([1, 2, 3, 4, 5], True),
        ([0, 1, 5], False),
        ([1.0, 2.0, 3.0], True),
        (range(10), True),
        (range(10, 0, -1), False)
    ]
)
def test_check_if_numbers_are_consecutive(input, expected_result, rp_logger):
    """
    Test the function 'check_if_numbers_are_consecutive'
    """
    rp_logger.info(
        "Test the function 'check_if_numbers_are_consecutive'"
    )
    actual = check_if_numbers_are_consecutive(input)
    assert actual is expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_empty_dict(rp_logger):
    """
    Test the function 'flatten' if the input is the empty dictionary
    """
    rp_logger.info(
        "Test the function 'flatten' if the input is the empty dictionary"
    )
    actual = flatten({})
    assert actual == dict()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_flat_dict(rp_logger):
    """
    Test the function 'flatten' if the input is the flat dictionary
    """
    rp_logger.info(
        "Check the function 'flatten' if the input is the flat dictionary"
    )
    dic = {"a": "1", "b": "2", "c": 3}
    expected = dic
    actual = flatten(dic)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_with_one_key_dict(rp_logger):
    """
    Test the function 'flatten' if the input is the dictionary with one key that is a dictionary
    """
    rp_logger.info(
        "Check the function 'flatten' if the input is the dictionary "
        "with one key that is a dictionary"
    )
    dic = {
        "a": "1",
        "b": "2",
        "c": {
            "c1": "3",
            "c2": "4"
        }
    }
    expected = {"a": "1", "b": "2", "c_c1": "3", "c_c2": "4"}
    actual = flatten(dic)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_with_one_key_value_utf8(rp_logger):
    """
    Test the `flatten` function with a dictionary containing UTF-8 encoded string keys and values.
    The test uses a sample dictionary with both ASCII and UTF-8 character.
    """
    rp_logger.info(
        "Test the `flatten` function with a dictionary containing UTF-8 encoded "
        "string keys and values"
    )
    dic = {
        "a": "1",
        u"ñ": u"áéö",
        "c": {u"c1": "3", "c2": "4"}
    }
    expected = {"a": "1", u"ñ": u"áéö", "c_c1": "3", "c_c2": "4"}
    actual = flatten(dic)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_with_one_key_utf8(rp_logger):
    """
    Test the `flatten` function with a nested dictionary containing UTF-8 encoded keys
    """
    rp_logger.info(
        "Test the `flatten` function with a nested dictionary containing UTF-8 encoded keys"
    )
    a = {u"eñe": 1}
    info = dict(info=a)
    expected = {u"info_{}".format(u"eñe"): 1}
    actual = flatten(info)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_with_custom_separator(rp_logger):
    """
    Test the 'flatten' function by providing the custom separator
    """
    rp_logger.info(
        "Test the 'flatten' function by providing the custom separator"
    )
    dic = {
        "a": "1",
        "b": "2",
        "c": {"c1": "3", "c2": "4"}
    }
    expected = {"a": "1", "b": "2", "c*c1": "3", "c*c2": "4"}
    actual = flatten(dic, "*")
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_dict_with_key_list(rp_logger):
    """
    Test the 'flatten' function with a nested dictionary containing a list as a value
    """
    rp_logger.info(
        "Test the 'flatten' function with a nested dictionary containing a list as a value"
    )
    dic = {
        "a": 1,
        "b": [{"c": [2, 3]}]
    }
    expected = {"a": 1, "b_0_c_0": 2, "b_0_c_1": 3}
    actual = flatten(dic)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_with_list_and_dict(rp_logger):
    """
    Test the 'flatten' function with a nested dictionary
    containing a dictionary and a list as a value
    """
    rp_logger.info(
        "Test the 'flatten' function with a nested dictionary "
        "containing a dictionary and a list as a value"
    )
    dic = {
        "a": 1,
        "b": 2,
        "c": [{"d": ["2", 3, 4], "e": [{"f": 1, "g": 2}]}]
    }
    expected = {
        "a": 1,
        "b": 2,
        "c_0_d_0": "2",
        "c_0_d_1": 3,
        "c_0_d_2": 4,
        "c_0_e_0_f": 1,
        "c_0_e_0_g": 2
    }
    actual = flatten(dic)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_with_empty_list_and_dict(rp_logger):
    """
    Test the 'flatten' function with a nested dictionary
    containing an empty list as a value, and an empty dictionary as a value
    """
    rp_logger.info(
        "Test the 'flatten' function with a nested dictionary "
        "containing an empty list as a value, and an empty dictionary as a value"
    )
    dic = {
        "a": {},
        "b": [],
        "c": "",
        "d": None,
        "e": [{"f": [], "g": [{"h": {}, "i": [], "j": "", "k": None}]}]
        }
    expected = {
        "a": {},
        "b": [],
        "c": "",
        "d": None,
        "e_0_f": [],
        "e_0_g_0_h": {},
        "e_0_g_0_i": [],
        "e_0_g_0_j": "",
        "e_0_g_0_k": None
    }
    actual = flatten(dic)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_with_tuple(rp_logger):
    """
    Test the 'flatten' function with a nested dictionary
    containing a nested tuple as a value
    """
    rp_logger.info(
        "Test the 'flatten' function with a nested dictionary "
        "containing a nested tuple as a value"
    )
    dic = {
        "a": 1,
        "b": ({"c": (2, 3)},)
    }
    expected = {"a": 1, "b_0_c_0": 2, "b_0_c_1": 3}
    actual = flatten(dic)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_with_empty_tuple(rp_logger):
    """
    Test the 'flatten' function with a nested dictionary
    containing an empty tuple as a value
    """
    rp_logger.info(
        "Test the 'flatten' function with a nested dictionary "
        "containing an empty tuple as a value"
    )
    dic = {
        "a": 1,
        "b": ({"c": ()},)
    }
    expected = {"a": 1, "b_0_c": ()}
    actual = flatten(dic)
    assert actual == expected


def test_unflatten_no_list(rp_logger):
    """
    Test the 'unflatten' function without an expected list as a value in the expected result
    """
    rp_logger.info(
        "Test the 'unflatten' function without an expected list as a value in the expected result"
    )
    dic = {
        "a": 1,
        "b_a": 2,
        "b_b": 3,
        "c_a_b": 5
    }
    expected = {
        "a": 1,
        "b": {"a": 2, "b": 3},
        "c": {"a": {"b": 5}}
    }
    actual = unflatten(dic)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "input, expected_result",
    [
        (
            {
                "a": 1,
                "b_0": 1,
                "b_1": 2,
                "c_a": "a",
                "c_b_0": 1,
                "c_b_1": 2,
                "c_b_2": 3
            },
            {
                "a": 1,
                "b": [1, 2],
                "c": {"a": "a", "b": [1, 2, 3]}
            }
        ),
        (
            {"a": 1, "b_0": 5},
            {"a": 1, "b": [5]}
        )
    ]
)
def test_unflatten_with_list(input, expected_result, rp_logger):
    """
    Test the 'unflatten_list' function with an expected list as a value in the expected result
    """
    rp_logger.info(
        "Test the 'unflatten_list' function "
        "with an expected list as a value in the expected result"
    )
    actual = unflatten_list(input)
    assert actual == expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_unflatten_with_list_custom_separator(rp_logger):
    """
    Test the 'unflatten_list' function by providing with a custom separator
    """
    rp_logger.info(
        "Test the 'unflatten_list' function by providing with a custom separator"
    )
    dic = {
        "a:b": "str0",
        "c:0:d:0:e": "str1",
        "c:1:d:0:e": "str4",
        "c:1:f": "str5",
        "c:0:f": "str2",
        "c:1:g": "str6",
        "c:0:g": "str3",
        "h:d:0:e": "str7",
        "h:i:0:f": "str8",
        "h:i:0:g": "str9"
    }
    expected = {
        "a": {"b": "str0"},
        "c": [
            {
                "d": [{"e": "str1"}],
                "f": "str2",
                "g": "str3"
            },
            {
                "d": [{"e": "str4"}],
                "f": "str5",
                "g": "str6"
            }
        ],
        "h": {
            "d": [{"e": "str7"}],
            "i": [{"f": "str8", "g": "str9"}]
        }
    }
    actual = unflatten_list(dic, ":")
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "nested_list",
    [
        [2.352791, 48.830825],
        [[[14.5, 32.8308], [6.792, 8.83]]],
        [
            [[14.5, 32.8308], [6.792, 8.83]],
            [[15.9, 35.9308], [16.892, 10.83]]
        ],
        [[2.352791, 48.830825], [5.352792, 19.830826]],
        [
            [[[14.5, 32.8308], [6.792, 8.83]], [[24.5, 12.8308], [36.792, 18.83]]],
            [[[37.5, 36.8308], [18.792, 28.83]], [[44.5, 12.8308], [66.792, 58.83]]]
        ],
    ]
)
def test_unflatten_with_list_nested(nested_list, rp_logger):
    """
    Test the 'unflatten_list' function with a nested list as a value
    with different levels of nesting
    """
    rp_logger.info(
        "Test the 'unflatten_list' function with a nested list as a value "
        "with different levels of nesting"
    )
    dic = {"type": "Point", "coordinates": nested_list}
    dic_flatten = flatten(dic)
    actual = unflatten_list(dic_flatten)
    assert actual == dic
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "input, expected_result",
    [
        (
            {
                "coordinates_0": 11.678678,
                "coordinates_1": 45.678678
            },
            {"coordinates": [11.678678, 45.678678]}
        ),
        (
            {
                "coordinates_0_0": 14.5,
                "coordinates_0_1": 32.8308,
                "coordinates_1_0": 56.7897,
                "coordinates_1_1": 8.833
            },
            {"coordinates": [[14.5, 32.8308], [56.7897, 8.833]]}
        ),
        (
            {
                "coordinates_0": 14.5,
                "coordinates_1": 32.8308,
                "coordinates_0_0": 14.6786,
                "coordinates_0_0_0": 35.8308,
                "coordinates_0_0_1": 56.7897
            },
            {"coordinates": [[[35.8308, 56.7897]], 32.8308]}
        ),
        (
            {
                "coordinates_1": 25.4308,
                "coordinates_0_1": 14.6786,
                "coordinates_0_0_0": 35.8308,
                "coordinates_0_0_1": 56.7897
            },
            {"coordinates": [[[35.8308, 56.7897], 14.6786], 25.4308]}
        ),
        (
            {
                "coordinates_1": 25.4308,
                "coordinates_1_0": 47.46018,
                "coordinates_1_1": 25.4368,
                "coordinates_0_1": 14.6786,
                "coordinates_0_0_0": 35.8308,
                "coordinates_0_0_1": 56.7897
            },
            {"coordinates": [[[35.8308, 56.7897], 14.6786], [47.46018, 25.4368]]}
        )
    ]
)
def test_unflatten_with_list_with_diff_nested_levels(input, expected_result, rp_logger):
    """
    Test the 'unflatten_list' function with a nested list as a value
    with different levels of nesting and the custom separator
    """
    rp_logger.info(
        "Test the 'unflatten_list' function with a nested list as a value "
        "with different levels of nesting and the custom separator"
    )
    actual = unflatten_list(input, separator='_')
    assert actual == expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_unflatten_with_list_nested_with_diff_levels(rp_logger):
    """
    Test the 'unflatten_list' function with different levels of nesting
    """
    rp_logger.info(
        "Test the 'unflatten_list' function with different levels of nesting"
    )
    dic = {"a": [[{"b": 1}], [{"d": 1}]]}
    dic_flatten = flatten(dic)
    actual = unflatten_list(dic_flatten)
    assert actual == dic
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_unflatten_with_list_issue15():
    """https://github.com/amirziai/flatten/issues/15"""
    dic = {
        "Required": {
            "a": "1",
            "b": ["1", "2", "3"],
            "c": {
                "d": {
                    "e": [
                        [{"s1": 1}, {"s2": 2}],
                        [{"s3": 1}, {"s4": 2}]
                    ]
                }
            },
            "f": ["1", "2"]},
        "Optional": {"x": "1", "y": ["1", "2", "3"]}
    }
    dic_flatten = flatten(dic)
    actual = unflatten_list(dic_flatten)
    assert actual == dic


def test_unflatten_with_list_issue31():
    """https://github.com/amirziai/flatten/issues/31"""
    dic = {
        "testdict": {
            "seconddict": [
                ["firstvalue", "secondvalue"],
                ["thirdvalue", "fourthvalue"]
            ]
        }
    }

    dic_flatten = flatten(dic)
    actual = unflatten_list(dic_flatten)
    assert actual == dic


def test_unflatten_with_key_loss_issue51(rp_logger):
    """https://github.com/amirziai/flatten/issues/51"""
    dic = {
        "a": 1,
        "a_b": 2,
        "a_c.d": 3,
        "a_c.e": 4
    }
    expected = {
        "a": 1,
        "a_b": 2,
        "a_c": {"d": 3, "e": 4}
    }
    actual = unflatten(dic, '.')
    assert actual == expected


def test_flatten_preserve_lists_issue43_nested():
    """https://github.com/amirziai/flatten/issues/43"""
    dic = {
        "a": {"a": ["x0", "x1", "x2"]},
        "b": {"b": "foo", "c": "bar"},
        "c": {"c": [
            {"foo": 2, "bar": 6, "baz": ["n1", "n2", "n3", "n1.1", "n2.2"]},
            {"foo": 5, "bar": 7, "baz": ["n4", "n5", "n6"]},
            {"foo": float('nan')},
            {"foo": 100},
        ]},
        "d": {"g": 10},
        "f": {
                "h": 100,
                "gar": [
                    {
                        "gup": 200,
                        "garp": [
                            {"gu": 300, "gat": ["f7", "f8"]},
                            {"gu": 800, "gat": ["f9", "f10", "f11"]}
                        ]
                    }
                ]
            }
        }
    actual = flatten_preserve_lists(dic, max_depth=100, max_list_index=30)

    expected = [
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n3",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n1.1",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10.
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n2.2",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 6,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 2,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 5,
            "d": 10,
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n4",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n5",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f7",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f8",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f9",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f10",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": None,
            "c_c_baz": "n6",
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": 7,
            "f_gar_garp_gat": "f11",
            "c_c_foo": 5,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f7",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f7",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f7",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f8",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f8",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 300,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f8",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f9",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f9",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f9",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f10",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f10",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f10",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x0",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f11",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x1",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f11",
            "c_c_foo": None,
            "d": 10
        },
        {
            "a_a": "x2",
            "f_h": 100,
            "c_c": 100,
            "c_c_baz": None,
            "b_b": "foo",
            "b_c": "bar",
            "f_gar_garp_gu": 800,
            "f_gar_gup": 200,
            "c_c_bar": None,
            "f_gar_garp_gat": "f11",
            "c_c_foo": None,
            "d": 10
        }
    ]
    assert actual == expected


def test_flatten_preserve_lists_issue43():
    """https://github.com/amirziai/flatten/issues/43"""
    dic = {
        "a": {"a": ["x0", "x1", "x2"]},
        "b": {"b": "foo", "c": "bar"},
        "c": {"c": [
            {"foo": 2, "bar": 6, "baz": [
                    "n1", "n2", "n3", "n1.1", "n2.2"]},
            {"foo": 5, "bar": 7, "baz": ["n4", "n5", "n6"]},
            {"foo": float('nan')},
            {"foo": 100},
        ]},
        "d": {"g": 10}
    }
    expected = [
            {"a_a": "x0",
             "c_c_foo": 2,
             "c_c_baz": "n1",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x1",
             "c_c_foo": 2,
             "c_c_baz": "n1",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x2",
             "c_c_foo": 2,
             "c_c_baz": "n1",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x0",
             "c_c_foo": 2,
             "c_c_baz": "n2",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x1",
             "c_c_foo": 2,
             "c_c_baz": "n2",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x2",
             "c_c_foo": 2,
             "c_c_baz": "n2",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x0",
             "c_c_foo": 2,
             "c_c_baz": "n3",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x1",
             "c_c_foo": 2,
             "c_c_baz": "n3",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x2",
             "c_c_foo": 2,
             "c_c_baz": "n3",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x0",
             "c_c_foo": 2,
             "c_c_baz": "n1.1",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x1",
             "c_c_foo": 2,
             "c_c_baz": "n1.1",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x2",
             "c_c_foo": 2,
             "c_c_baz": "n1.1",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x0",
             "c_c_foo": 2,
             "c_c_baz": "n2.2",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x1",
             "c_c_foo": 2,
             "c_c_baz": "n2.2",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x2",
             "c_c_foo": 2,
             "c_c_baz": "n2.2",
             "c_c_bar": 6,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x0",
             "c_c_foo": 5,
             "c_c_baz": "n4",
             "c_c_bar": 7,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x1",
             "c_c_foo": 5,
             "c_c_baz": "n4",
             "c_c_bar": 7,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x2",
             "c_c_foo": 5,
             "c_c_baz": "n4",
             "c_c_bar": 7,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x0",
             "c_c_foo": 5,
             "c_c_baz": "n5",
             "c_c_bar": 7,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x1",
             "c_c_foo": 5,
             "c_c_baz": "n5",
             "c_c_bar": 7,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x2",
             "c_c_foo": 5,
             "c_c_baz": "n5",
             "c_c_bar": 7,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x0",
             "c_c_foo": 5,
             "c_c_baz": "n6",
             "c_c_bar": 7,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x1",
             "c_c_foo": 5,
             "c_c_baz": "n6",
             "c_c_bar": 7,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x2",
             "c_c_foo": 5,
             "c_c_baz": "n6",
             "c_c_bar": 7,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": None},
            {"a_a": "x0",
             "c_c_foo": None,
             "c_c_baz": None,
             "c_c_bar": None,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             "c_c": 100},
            {"a_a": "x1",
             "c_c_foo": None,
             "c_c_baz": None,
             "c_c_bar": None,
             "b_b": "foo",
             "d": 10,
             "b_c": "bar",
             'c_c': 100},
            {'a_a': 'x2',
             'c_c_foo': None,
             'c_c_baz': None,
             'c_c_bar': None,
             'b_b': 'foo',
             'd': 10,
             'b_c': 'bar',
             'c_c': 100}]

    actual = flatten_preserve_lists(dic, max_list_index=50, max_depth=10)
    assert actual == expected


def test_unflatten_with_list_deep(rp_logger):
    """
    Test the function 'unflatten_list' by providing the dictionary with a deep nested list
    """
    rp_logger.info(
        "Test the function 'unflatten_list' by providing the dictionary with a deep nested list"
    )
    dic = {'a': [
            {'b': [{'c': [{'a': 5, 'b': {'a': [1, 2, 3]}, 'c': {'x': 3}}]}]}]}
    dic_flatten = flatten(dic)
    actual = unflatten_list(dic_flatten)
    assert actual == dic
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_flatten_ignore_keys(rp_logger):
    """
    Test the function 'flatten' with ignoring a set of root keys for processing
    """
    rp_logger.info(
        "Test the function 'flatten' with ignoring a set of root keys for processing"
    )
    dic = {
        "a": {"a": [1, 2, 3]},
        "b": {"b": "foo", "c": "bar"},
        "c": {"c": [{"foo": 5, "bar": 6, "baz": [1, 2, 3]}]}
    }
    expected = {
        "a_a_0": 1,
        "a_a_1": 2,
        "a_a_2": 3
    }
    actual = flatten(dic, root_keys_to_ignore={"b", "c"})
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_replace_separators_none(rp_logger):
    """
    Test the 'flatten' function with the default parameter 'replace_separators' equals to 'None'
    """
    rp_logger.info(
        "Test the 'flatten' function with the default parameter 'replace_separators' "
        "equals to 'None'"
    )
    dic = {
        "a_with_separator": {"b": [1, 2, 3]},
    }
    expected = {
        "a_with_separator_b_0": 1,
        "a_with_separator_b_1": 2,
        "a_with_separator_b_2": 3
    }
    actual = flatten(dic)
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_replace_separators_remove(rp_logger):
    """
    Test the 'flatten' function with the parameter 'replace_separators' equals to the empty string
    """
    rp_logger.info(
        "Test the 'flatten' function with the parameter 'replace_separators' "
        "equals to the empty string"
    )
    dic = {
        "a_with_separator": {"b": [1, 2, 3]},
    }
    expected = {
        "awithseparator_b_0": 1,
        "awithseparator_b_1": 2,
        "awithseparator_b_2": 3
    }
    actual = flatten(dic, replace_separators="")
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_replace_separators_something(rp_logger):
    """
    Test the 'flatten' function with the parameter 'replace_separators' equals
    to the custom delimeter
    """
    rp_logger.info(
        "Test the 'flatten' function with the parameter 'replace_separators' "
        "equals to the custom delimeter"
    )
    dic = {
        "a_with_separator": {"b": [1, 2, 3]},
    }
    expected = {
        "a.with.separator_b_0": 1,
        "a.with.separator_b_1": 2,
        "a.with.separator_b_2": 3
    }
    actual = flatten(dic, replace_separators=".")
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_replace_separators_nested(rp_logger):
    """
    Test the 'flatten' function with the parameter 'replace_separators' equals to the empty string
    """
    rp_logger.info(
        "Test the 'flatten' function with the parameter 'replace_separators' "
        "equals to the empty string"
    )
    dic = {
        "a_with_separator": {'b_with_separator': [1, 2, 3]},
    }
    expected = {
        "awithseparator_bwithseparator_0": 1,
        "awithseparator_bwithseparator_1": 2,
        "awithseparator_bwithseparator_2": 3
    }
    actual = flatten(dic, replace_separators="")
    assert actual == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("value, expected_result", [
    (
        '{"key_1": "value_1", "key_2": "value_2"}',
        {
            'flattened_data': {'key_1': 'value_1', 'key_2': 'value_2'},
            'original_data': None
        }
    ),
    (
        b'{"key_1": "value_1", "key_2": "value_2"}',
        {
            'flattened_data': {'key_1': 'value_1', 'key_2': 'value_2'},
            'original_data': None
        }
    ),
    (
        '''
        {
        "member_1": {"name": "John", "occupation": "doctor"},
        "member_2": {"name": "Jane", "occupation": "manager", "department": "financial"}
        }
        ''',
        {
            "flattened_data": {
                "member_1.name": "John",
                "member_1.occupation": "doctor",
                "member_2.department": "financial",
                "member_2.name": "Jane",
                "member_2.occupation": "manager"
            },
            "original_data": None
        }
    ),
    (
        b'''
        {
        "member_1": {"name": "John", "occupation": "doctor"},
        "member_2": {"name": "Jane", "occupation": "manager", "department": "financial"}
        }
        ''',
        {
            "flattened_data": {
                "member_1.name": "John",
                "member_1.occupation": "doctor",
                "member_2.department": "financial",
                "member_2.name": "Jane",
                "member_2.occupation": "manager"
            },
            "original_data": None
        }
    ),
    (
        bytearray(b'''
        {
        "member_1": {"name": "John", "occupation": "doctor"},
        "member_2": {"name": "Jane", "occupation": "manager", "department": "financial"}
        }
        '''),
        {
            "flattened_data": {
                "member_1.name": "John",
                "member_1.occupation": "doctor",
                "member_2.department": "financial",
                "member_2.name": "Jane",
                "member_2.occupation": "manager"
            },
            "original_data": None
        }
    ),
    ("[1, 2, 3, 4, 5]", {"flattened_data": {}, "original_data": '[1, 2, 3, 4, 5]'}),
    ("Not a JSON string", {"flattened_data": {}, "original_data": "Not a JSON string"}),
    (12345, {"flattened_data": {}, "original_data": 12345}),
    (123.45, {"flattened_data": {}, "original_data": 123.45}),
    (np.nan, {"flattened_data": {}, "original_data": np.nan}),
    (None, {"flattened_data": {}, "original_data": None}),
    (True, {"flattened_data": {}, "original_data": True}),
    (False, {"flattened_data": {}, "original_data": False}),
])
def test_safe_flatten(value, expected_result, rp_logger):
    rp_logger.info("Test the method 'safe_flatten'")
    assert safe_flatten(value) == expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)
