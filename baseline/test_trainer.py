import os
import warnings

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from trainer.trainer import Trainer

from srai.datasets import AirbnbMulticity
from srai.embedders import Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.regionalizers import H3Regionalizer

os.environ["HF_access_token"] = ""
airbnb_multicity = AirbnbMulticity()
airbnb_multicity_gdf = airbnb_multicity.load()


resolution = 8
gdf = airbnb_multicity_gdf.loc[airbnb_multicity_gdf["city"] == "paris"]
regionalizer = H3Regionalizer(resolution=resolution)
regions = regionalizer.transform(gdf)

loader = OSMPbfLoader()
# works faster, but for smaller resolution there might eb an error (empty features), so better to use it on gdfb
# features = loader.load(regions, HEX2VEC_FILTER)
features = loader.load(gdf, HEX2VEC_FILTER)

joiner = IntersectionJoiner()
joint = joiner.transform(regions, features)

neighbourhood = H3Neighbourhood(regions)
embedder_hidden_sizes = [150, 100, 50]
embedder = Hex2VecEmbedder(embedder_hidden_sizes)

device = "cuda" if torch.cuda.is_available() else "cpu"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    embeddings = embedder.fit_transform(
        regions,
        features,
        joint,
        neighbourhood,
        trainer_kwargs={"max_epochs": 5, "accelerator": device},
        batch_size=100,
    )

embeddings_size = embeddings.values.shape[1]


def concat_columns(row):
    return np.concatenate([np.atleast_1d(val) for val in row.values])


embeddings["vector_embedding"] = embeddings.apply(concat_columns, axis=1)

joined_gdf = gpd.sjoin(gdf, regions, how="left", op="within")
joined_gdf.rename(columns={"index_right": "h3_index"}, inplace=True)
average_hex_prices = joined_gdf.groupby("h3_index")["price"].mean()
embeddings["h3"] = embeddings.index
merged_gdf = embeddings.merge(
    average_hex_prices, how="inner", left_on="region_id", right_on="h3_index"
)

model = nn.Sequential(
    nn.Linear(embeddings_size, 225),
    nn.Sigmoid(),
    nn.Dropout(0.2),
    nn.Linear(225, 100),
    nn.Sigmoid(),
    nn.Dropout(0.2),
    nn.Linear(100, 50),
    nn.Sigmoid(),
    nn.Dropout(0.2),
    nn.Linear(50, 25),
    nn.Sigmoid(),
    nn.Dropout(0.2),
    nn.Linear(25, 1),
    nn.ReLU(),
)
loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_df, eval_df = train_test_split(merged_gdf, train_size=0.8, shuffle=True)

train_args = {
    "epochs": 50,
    "batch_size": 100,
    "input_col": "vector_embedding",
    "labels": "price",
    "device": device,
}

trainer = Trainer(
    model=model,
    args=train_args,
    train_df=train_df,
    eval_df=eval_df,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
model, _, _ = trainer.train()
