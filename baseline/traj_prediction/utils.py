import torch
from shapely import LineString
import plotly.graph_objs as go
import random

# TODO add fun from notebooks


def haversine_loss2(pred, target, epsSq = 1.e-13, epsAs = 1.e-7):   # add optional epsilons to avoid singularities
    # print ('haversine_loss: epsSq:', epsSq, ', epsAs:', epsAs)
    lat1, lon1 = torch.split(pred, 1, dim=1)
    lat2, lon2 = torch.split(target, 1, dim=1)
    r = 6371  # Radius of Earth in kilometers
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi, delta_lambda = torch.deg2rad(lat2-lat1), torch.deg2rad(lon2-lon1)
    a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
    # return tensor.mean(2 * r * torch.asin(torch.sqrt(a)))
    # "+ (1.0 - a**2) * epsSq" to keep sqrt() away from zero
    # "(1.0 - epsAs) *" to keep asin() away from plus-or-minus one
    a_clamped = torch.clamp(a, -1.0 + epsAs, 1.0 - epsAs)
    return torch.mean(2 * r * torch.asin(torch.sqrt(a_clamped + (1.0 - a_clamped**2) * epsSq)))

def sample_traj(loader, n):
    loader = list(loader)
    batch = random.sample(loader, n)
    res = []
    for X, y in batch:
        traj = []
        for xi in X[0]:
            traj.append([xi[0].item(), xi[1].item()])
        res.append(traj)
    return res

def plot_trajectories(trajectories):
    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    if isinstance(trajectories[0], LineString):
        trajectories = [list(trajectory.coords) for trajectory in trajectories]

    fig = go.Figure()
    for trajectory in trajectories:
        lon = [point[0] for point in trajectory]
        lat = [point[1] for point in trajectory]

        fig.add_trace(go.Scattermapbox(
            mode="markers+lines",
            lon=lon,
            lat=lat,
            marker={'size': 10}
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=10,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    fig.show()