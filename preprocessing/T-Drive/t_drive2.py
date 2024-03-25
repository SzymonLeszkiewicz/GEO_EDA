import datasets
import pyarrow as pa
import pyarrow.parquet as pq

_URLS = {"tdrive": "https://huggingface.co/datasets/kraina/t_drive/resolve/main/data/train.parquet"}

_DESCRIPTION = '''This dataset contains the GPS trajectories of 10,357 taxis during the period of Feb. 2 to Feb. 8, 2008
within Beijing. The total number of points in this dataset is about 15 million and the total distance of
the trajectories reaches to 9 million kilometers.'''

CITATION = '''[1] Jing Yuan, Yu Zheng, Xing Xie, and Guangzhong Sun. Driving with knowledge from the physical world.
In The 17th ACM SIGKDD international conference on Knowledge Discovery and Data mining, KDD
’11, New York, NY, USA, 2011. ACM.
[2] Jing Yuan, Yu Zheng, Chengyang Zhang, Wenlei Xie, Xing Xie, Guangzhong Sun, and Yan Huang. Tdrive: driving directions based on taxi trajectories. In Proceedings of the 18th SIGSPATIAL International
Conference on Advances in Geographic Information Systems, GIS ’10, pages 99–108, New York, NY, USA,
2010. ACM.'''


class TdriveDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig """

    def __init__(self, data_url, **kwargs):
        """BuilderConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TdriveDatasetConfig, self).__init__(**kwargs)
        self.data_url = data_url


class TdriveDataset(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = TdriveDatasetConfig
    DEFAULT_CONFIG_NAME = "tdrive"
    BUILDER_CONFIGS = [
        TdriveDatasetConfig(
            name="tdrive",
            description="todo",
            data_url="https://huggingface.co/datasets/kraina/t_drive/resolve/main/data/train.parquet"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "taxi_id": datasets.Value("string"),
                    "date_time": datasets.Sequence(datasets.Value("string")),
                    "lon": datasets.Sequence(datasets.Value("float32")),
                    "lat": datasets.Sequence(datasets.Value("float32")),
                    "arrays_geometry": datasets.Sequence(datasets.Sequence(datasets.Value(dtype="float64")))

                }
            ),
            supervised_keys=None)

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        # files = _URLS[self.config.name]
        downloaded_files = dl_manager.download(self.config.data_url)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath': downloaded_files})
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_tables(self, filepath):
        with open(filepath, mode="rb") as f:
            parquet_file = pq.ParquetFile(source=filepath)
            for batch_idx, record_batch in enumerate(parquet_file.iter_batches()):
                pa_table = pa.Table.from_batches([record_batch])
                yield f"{batch_idx}", pa_table
