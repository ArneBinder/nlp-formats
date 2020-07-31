import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import path

import nlp


@dataclass
class BratConfig(nlp.BuilderConfig):
    """BuilderConfig for BRAT."""

    ann_file_extension: str = 'ann'
    txt_file_extension: str = 'txt'


class AbstractBrat(nlp.GeneratorBasedBuilder, ABC):
    BUILDER_CONFIG_CLASS = BratConfig

    def _info(self):
        return nlp.DatasetInfo(
            features=nlp.Features({
                "context": nlp.Value("string"),
                "spans": nlp.Sequence({
                    "id": nlp.Value("string"),
                    "type": nlp.Value("string"),
                    "locations": nlp.Sequence({
                        "start": nlp.Value("int32"),
                        "end": nlp.Value("int32"),
                    }),
                    "text": nlp.Value("string"),
                }),
                "relations": nlp.Sequence({
                    "id": nlp.Value("string"),
                    "type": nlp.Value("string"),
                    "arguments": nlp.Sequence({
                        "type": nlp.Value("string"),
                        "target": nlp.Value("string")
                    })
                }),
                "equivalence_relations": nlp.Sequence({
                    "type": nlp.Value("string"),
                    "targets": nlp.Sequence(nlp.Value("string")),
                }),
                "events": nlp.Sequence({
                    "id": nlp.Value("string"),
                    "type": nlp.Value("string"),
                    "trigger": nlp.Value("string"),
                    "arguments": nlp.Sequence({
                        "type": nlp.Value("string"),
                        "target": nlp.Value("string")
                    })
                }),
                "attributions": nlp.Sequence({
                    "id": nlp.Value("string"),
                    "type": nlp.Value("string"),
                    "target": nlp.Value("string"),
                    "value": nlp.Value("string"),

                }),
                "normalizations": nlp.Sequence({
                    "id": nlp.Value("string"),
                    "type": nlp.Value("string"),
                    "target": nlp.Value("string"),
                    "resource_id": nlp.Value("string"),
                    "entity_id": nlp.Value("string"),
                }),
                "notes": nlp.Sequence({
                    "id": nlp.Value("string"),
                    "type": nlp.Value("string"),
                    "target": nlp.Value("string"),
                    "note": nlp.Value("string"),
                }),
            })
        )

    @abstractmethod
    def _split_generators(self, dl_manager):
        pass

    @staticmethod
    def _get_location(location_string):
        parts = location_string.split(' ')
        assert len(parts) == 2, f'Wrong number of entries in location string. Expected 2, but found: {parts}'
        return {'start': int(parts[0]), 'end': int(parts[1])}

    @staticmethod
    def _get_span_annotation(annotation_line):
        """
        example input:
        T1	Organization 0 4	Sony
        """

        _id, remaining, text = annotation_line.split('\t', maxsplit=2)
        _type, locations = remaining.split(' ', maxsplit=1)
        return {
            'id': _id,
            'text': text,
            'type': _type,
            'locations': [AbstractBrat._get_location(loc) for loc in locations.split(';')]
        }

    @staticmethod
    def _get_event_annotation(annotation_line):
        """
        example input:
        E1	MERGE-ORG:T2 Org1:T1 Org2:T3
        """
        _id, remaining = annotation_line.strip().split('\t')
        args = [dict(zip(['type', 'target'], a.split(':'))) for a in remaining.split(' ')]
        return {
            'id': _id,
            'type': args[0]['type'],
            'trigger': args[0]['target'],
            'arguments': args[1:]
        }

    @staticmethod
    def _get_relation_annotation(annotation_line):
        """
        example input:
        R1	Origin Arg1:T3 Arg2:T4
        """

        _id, remaining = annotation_line.strip().split('\t')
        _type, remaining = remaining.split(' ', maxsplit=1)
        args = [dict(zip(['type', 'target'], a.split(':'))) for a in remaining.split(' ')]
        return {
            'id': _id,
            'type': _type,
            'arguments': args
        }

    @staticmethod
    def _get_equivalence_relation_annotation(annotation_line):
        """
        example input:
        *	Equiv T1 T2 T3
        """
        _, remaining = annotation_line.strip().split('\t')
        parts = remaining.split(' ')
        return {
            'type': parts[0],
            'targets': parts[1:]
        }

    @staticmethod
    def _get_attribute_annotation(annotation_line):
        """
        example input (binary: implicit value is True, if present, False otherwise):
        A1	Negation E1
        example input (multi-value: explicit value)
        A2	Confidence E2 L1
        """

        _id, remaining = annotation_line.strip().split('\t')
        parts = remaining.split(' ')
        # if no value is present, it is implicitly "true"
        if len(parts) == 2:
            parts.append('true')
        return {
            'id': _id,
            'type': parts[0],
            'target': parts[1],
            'value': parts[2],
        }

    @staticmethod
    def _get_normalization_annotation(annotation_line):
        """
        example input:
        N1	Reference T1 Wikipedia:534366	Barack Obama
        """
        _id, remaining, text = annotation_line.split('\t', maxsplit=2)
        _type, target, ref = remaining.split(' ')
        res_id, ent_id = ref.split(':')
        return {
            'id': _id,
            'type': _type,
            'target': target,
            'resource_id': res_id,
            'entity_id': ent_id,
        }

    @staticmethod
    def _get_note_annotation(annotation_line):
        """
        example input:
        #1	AnnotatorNotes T1	this annotation is suspect
        """
        _id, remaining, note = annotation_line.split('\t', maxsplit=2)
        _type, target = remaining.split(' ')
        return {
            'id': _id,
            'type': _type,
            'target': target,
            'note': note,
        }

    @staticmethod
    def _read_annotation_file(filename):
        """
        reads a BRAT v1.3 annotations file (see https://brat.nlplab.org/standoff.html)
        """

        res = {
            'spans': [],
            'events': [],
            'relations': [],
            'equivalence_relations': [],
            'attributions': [],
            'normalizations': [],
            'notes': [],
        }

        with open(filename) as file:
            for line in file:
                if len(line) == 0:
                    continue
                ann_type = line[0]

                # strip away the new line character
                line = line[:-1]

                if ann_type == 'T':
                    res['spans'].append(AbstractBrat._get_span_annotation(line))
                elif ann_type == 'E':
                    res['events'].append(AbstractBrat._get_event_annotation(line))
                elif ann_type == 'R':
                    res['relations'].append(AbstractBrat._get_relation_annotation(line))
                elif ann_type == '*':
                    res['relations'].append(AbstractBrat._get_equivalence_relation_annotation(line))
                elif ann_type in ['A', 'M']:
                    res['attributions'].append(AbstractBrat._get_attribute_annotation(line))
                elif ann_type == 'N':
                    res['normalizations'].append(AbstractBrat._get_normalization_annotation(line))
                elif ann_type == '#':
                    res['notes'].append(AbstractBrat._get_note_annotation(line))
                else:
                    raise ValueError(f'unknown BRAT id type: {line[0]}. Annotation ids have to start with T (spans), '
                                     f'E (events), R (relations), A (attributions), or N (normalizations). See '
                                     f'https://brat.nlplab.org/standoff.html for the BRAT annotation file '
                                     f'specification.')
        return res

    def _generate_examples(self, files=None, directory=None):
        """ Read context (.txt) and annotation (.ann) files. """
        if files is None:
            assert directory is not None, 'If files is None, directory has to be provided, but it is also None.'
            _files = glob.glob(f"{directory}/*.{self.config.ann_file_extension}")
            files = [path.splitext(fn)[0] for fn in _files]

        for filename in files:
            basename = path.basename(filename)

            ann_fn = f'{filename}.{self.config.ann_file_extension}'
            brat_annotations = AbstractBrat._read_annotation_file(ann_fn)

            txt_fn = f'{filename}.{self.config.txt_file_extension}'
            txt_content = open(txt_fn).read()
            brat_annotations['context'] = txt_content

            yield basename, brat_annotations

