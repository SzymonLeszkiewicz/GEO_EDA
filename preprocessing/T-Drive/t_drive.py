'''this script is used to load the T-Drive dataset from the parquet files'''

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

_DESCRIPTION = '''This dataset contains the GPS trajectories of 10,357 taxis during the period of Feb. 2 to Feb. 8, 2008
within Beijing. The total number of points in this dataset is about 15 million and the total distance of
the trajectories reaches to 9 million kilometers.'''


class TDrive(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "taxi_id": datasets.Value("string"),
                    "date_time": datasets.Sequence(datasets.Value("string")),
                    "longitude": datasets.Sequence(datasets.Value("float32")),
                    "latitude": datasets.Sequence(datasets.Value("float32")),
                    "arrays_geometry": datasets.Sequence(datasets.Sequence(datasets.Value(dtype="float64")))

                }
            ),
            supervised_keys=None)

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": "data/train.parquet", "split": "train"}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": "data/test.parquet", "split": "test"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": "data/val.parquet", "split": "validation"}),

        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_tables(self, filepath):
        with open(filepath, mode="rb") as f:
            parquet_file = pq.ParquetFile(source=filepath)
            for batch_idx, record_batch in enumerate(parquet_file.iter_batches()):
                pa_table = pa.Table.from_batches([record_batch])
                yield f"{batch_idx}", pa_table
