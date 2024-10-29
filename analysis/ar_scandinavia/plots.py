"""Some plots."""

import numpy as np
import proplot as pplt  # type: ignore
import xarray as xr
from cartopy.mpl.geocollection import QuadMesh  # type: ignore


def cluster_plot(
    ar_vals: list[xr.DataArray],
    pr_vals: list[xr.DataArray],
    ar_ds: xr.Dataset,
    ardt_name: str,
    tot_pr: xr.Dataset,
    cluster_counts: np.ndarray,
    resample_id: str,
    n_clusters: int,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
) -> pplt.Figure:
    """Create a plot of AR/PR clusters."""
    tot_pr_vmax = tot_pr.tp.max().values
    n_timesteps = ar_ds.resample(time=resample_id).sum().time.shape[0]
    cluster_id = 1
    fig, axs = pplt.subplots(
        nrows=n_clusters,
        ncols=2,
        proj="nsper",
        proj_kw={"lon_0": 14, "lat_0": 65},
    )

    for cluster_id in range(n_clusters):
        # Get the AR frequency
        curr_cluster_counts = cluster_counts[cluster_id]

        ar_cm = (ar_vals[cluster_id] / curr_cluster_counts * 100).plot(
            ax=axs[cluster_id, 0],
            cmap="fire",
            add_colorbar=False,
            vmin=0,
            rasterized=True,
        )
        axs[cluster_id, 0].format(
            title=f"n: {curr_cluster_counts} [{curr_cluster_counts/n_timesteps*100:.2f}%]"
        )
        ar_cbar = axs[cluster_id, 0].colorbar(
            ar_cm, loc="right", label="Cluster relative AR frequency [%]"
        )
        ax_extra = ar_cbar.ax.twinx()
        ticks = ar_cbar.get_ticks()
        n_ticks = len(ticks)

        ar_frac_labels = ticks * curr_cluster_counts / n_timesteps

        ar_vmin = ar_cbar.vmin
        ar_vmax = ar_cbar.vmax

        ax_extra.format(
            ylocator=ticks,
            ytickminor=False,
            ylim=(ar_vmin, ar_vmax),
            yticklabels=[f"{x:.1f}" for x in ar_frac_labels],
            ylabel="Overall AR frequency [%]",
        )

        # Total precipitation fraction
        pr_cm = pr_vals[cluster_id].plot(
            ax=axs[cluster_id, 1],
            cmap="oslo_r",
            add_colorbar=False,
            vmin=0,
            rasterized=True,
        )

        pr_cbar = axs[cluster_id, 1].colorbar(
            pr_cm, loc="right", label="Cluster total precipitation [m]"
        )
        ax_extra = pr_cbar.ax.twinx()
        ticks = pr_cbar.get_ticks()
        n_ticks = len(ticks)

        if n_ticks > 6:
            ticks = ticks[::2]

        pr_frac_labels = ticks / tot_pr_vmax * 100
        pr_vmin = pr_cbar.vmin
        pr_vmax = pr_cbar.vmax
        ax_extra.format(
            ylocator=ticks,
            ytickminor=False,
            ylim=(pr_vmin, pr_vmax),
            yticklabels=[f"{x:.1f}" for x in pr_frac_labels],
            ylabel="Total precipitation fraction [%]",
        )

    # fig.colorbar(ar_cm, loc="top", label="AR frequency [%]", col=1)
    axs.format(
        coast=True,
        reso="med",
        lonlim=lon_slice,
        latlim=lat_slice,
        leftlabels=[f"Cluster {n+1}" for n in range(n_clusters)],
        toplabels=["AR frequency", "Total precipitation \ngiven AR cluster"],
        # toplabelpad="20mm",
    )
    fig.format(
        suptitle=f"AR and precipitation | ARDT: {ardt_name}, {resample_id}",
    )
    return fig
