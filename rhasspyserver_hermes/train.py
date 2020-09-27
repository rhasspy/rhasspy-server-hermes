"""Methods for training Rhasspy"""
import io
import logging
import typing
from pathlib import Path

import networkx as nx

import rhasspynlu
from rhasspynlu.jsgf import Expression, Word

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def sentences_to_graph(
    sentences_dict: typing.Dict[str, str],
    slots_dirs: typing.Optional[typing.List[Path]] = None,
    slot_programs_dirs: typing.Optional[typing.List[Path]] = None,
    replace_numbers: bool = True,
    language: str = "en",
    word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    add_intent_weights: bool = True,
) -> nx.DiGraph:
    """Transform sentences to an intent graph"""
    slots_dirs = slots_dirs or []
    slot_programs_dirs = slot_programs_dirs or []

    # Parse sentences and convert to graph
    with io.StringIO() as ini_file:
        # Join as single ini file
        for lines in sentences_dict.values():
            print(lines, file=ini_file)
            print("", file=ini_file)

        # Parse JSGF sentences
        intents = rhasspynlu.parse_ini(ini_file.getvalue())

    # Split into sentences and rule/slot replacements
    sentences, replacements = rhasspynlu.ini_jsgf.split_rules(intents)

    word_visitor: typing.Optional[
        typing.Callable[[Expression], typing.Union[bool, Expression]]
    ] = None

    if word_transform:
        # Apply transformation to words

        def transform_visitor(word: Expression):
            if isinstance(word, Word):
                assert word_transform
                new_text = word_transform(word.text)

                # Preserve case by using original text as substition
                if (word.substitution is None) and (new_text != word.text):
                    word.substitution = word.text

                word.text = new_text

            return word

        word_visitor = transform_visitor

    # Apply case/number transforms
    if word_visitor or replace_numbers:
        for intent_sentences in sentences.values():
            for sentence in intent_sentences:
                if replace_numbers:
                    # Replace number ranges with slot references
                    # type: ignore
                    rhasspynlu.jsgf.walk_expression(
                        sentence, rhasspynlu.number_range_transform, replacements
                    )

                if word_visitor:
                    # Do case transformation
                    # type: ignore
                    rhasspynlu.jsgf.walk_expression(
                        sentence, word_visitor, replacements
                    )

    # Load slot values
    slot_replacements = rhasspynlu.get_slot_replacements(
        intents,
        slots_dirs=slots_dirs,
        slot_programs_dirs=slot_programs_dirs,
        slot_visitor=word_visitor,
    )

    # Merge with existing replacements
    for slot_key, slot_values in slot_replacements.items():
        replacements[slot_key] = slot_values

    if replace_numbers:
        # Do single number transformations
        for intent_sentences in sentences.values():
            for sentence in intent_sentences:
                rhasspynlu.jsgf.walk_expression(
                    sentence,
                    lambda w: rhasspynlu.number_transform(w, language),
                    replacements,
                )

    # Convert to directed graph
    intent_graph = rhasspynlu.sentences_to_graph(
        sentences, replacements=replacements, add_intent_weights=add_intent_weights
    )

    return intent_graph, slot_replacements
