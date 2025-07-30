"""Results Viewer Util

Open Issues
    log_plot: bool - Double check if it works as intended
"""

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, display
from plotly.subplots import make_subplots

plotly.offline.init_notebook_mode()
display(
    HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    )
)


def plot_qps_recall(
    df,
    color_by="mhs",
    title="QPS over Recall of Hierarchy Layers",
    plot_all_datasets=False,
    plot_all_datasets_override=False,  # overrides other parameters
    suppress_fig_show=False,
    limit_to_top_layer=False,
    smooth_line=None,
    showlegend=False,
    showscale=False,
    label_text="layer",
    # filters: if empty nothing gets filtered out
    # explicit filtering, means implicit auto-select all if empty..
    filter_out_hlmhs=[],
    filter_out_k=[],
    filter_out_builder=[],
    filter_out_dataset=[],
    smooth_line_bottom_level=False,
    smooth_line_top_level=False,
    facet_col=None,
    facet_row=None,
    height=None,  # 500
    margins=[60, 40, 40, 40],  # [b,t,r,l]
    template="plotly",
    log_plot=False,
):
    """
    Plots QPS over Recall for a dataframe

    NOTE averages over each **layer** (recall and qps)
    """
    # suppressing warnings
    pd.options.mode.chained_assignment = None
    # backward-compatibility fix
    df = df.rename(columns={"data_set": "dataset"})

    df["dataset"] = df["dataset"].apply(lambda ds: translate_dataset_name(ds))

    if plot_all_datasets_override:
        plot_all_datasets, showlegend = True, True
        facet_col, facet_row = None, None
        filter_out_k, filter_out_hlmhs = [], []
        smooth_line, limit_to_top_layer = True, True
        smooth_line_bottom_level, smooth_line_top_level = False, False
        color_by = "builder_type"

    all_figs = []  # only for plot_all_datasets
    datasets = df["dataset"].unique()

    for dataset in datasets:
        if dataset in filter_out_dataset:
            continue
        if plot_all_datasets:
            df_sub_plot = df.loc[df["dataset"] == dataset]
        else:
            df_sub_plot = df
        title = f"{dataset}"

        df_plot = prepare_df_plot(
            df_sub_plot,
            limit_to_top_layer,
            filter_out_hlmhs=filter_out_hlmhs,
            filter_out_k=filter_out_k,
            filter_out_builder=filter_out_builder,
            filter_out_dataset=filter_out_dataset,
        )

        ks = "".join([f"{e}," for e in df_plot["k"].unique()])[:-1]
        mhss = "".join([f"{e}," for e in df_plot["mhs"].unique()])[:-1]
        yaxis_title = "QPS"
        xaxis_title = f"k={ks}@ef=[{mhss}]-Recall"
        xaxis, yaxis = "recall", "qps"
        # convert chosen `color_by` column to str for discrete colors
        df_plot[color_by] = df_plot[color_by].astype(str)

        labels = {
            "builder_type": "Graph-Type",
            "mhs": "$ef_{bottom}$",
            "hlmhs": "$ef_{higher}$",
            "qps": "QPS",
            "recall": "1 - Recall" if log_plot else "Recall",
        }
        if facet_col != None or facet_row != None:
            labels["hlmhs"] = "ef-higher"

        if log_plot:
            # '100 - Recall' because recall-column is in percentage points
            df["recall"] = df["recall"].apply(lambda e: 100 - e)

        fig = px.scatter(
            df_plot,
            x=xaxis,
            y=yaxis,
            log_x=True if log_plot else False,
            log_y=True if log_plot else False,
            text=label_text,
            title=title,
            hover_data=[
                "layer",
                "mhs",
                "builder_type",
                "dataset",
                "build_time",
                "hlmhs",
            ],
            labels=labels,
            color=color_by,
            # color_discrete_sequence=["red", "green", "blue", "goldenrod", "magenta"],
            # symbol=color_by,
            # symbol_sequence=["circle-open", "circle", "circle-open-dot", "square"],
            facet_col=facet_col,
            facet_row=facet_row,
        )

        if smooth_line_bottom_level:
            df_local = df_plot.loc[df_plot["layer"] == int(1)]
            fig.add_traces(
                list(
                    px.line(
                        df_local,
                        x=xaxis,
                        y=yaxis,
                        line_shape="spline",
                        color_discrete_sequence=["gray"],
                        facet_col=facet_col,
                        facet_row=facet_row,
                    ).select_traces()
                )
            )

        if smooth_line_top_level:
            layers = df_plot["layer"].unique()
            top_layer = layers.max()
            df_local2 = df_plot.loc[df_plot["layer"] == top_layer]
            fig_local = px.line(
                df_local2,
                x=xaxis,
                y=yaxis,
                line_shape="spline",
                color_discrete_sequence=["gray"],
                facet_col=facet_col,
                facet_row=facet_row,
            )
            fig_local.update_traces(line=dict(width=1))  # might not have any effect
            fig.add_traces(list(fig_local.select_traces()))

        fig.update_coloraxes(showscale=showscale)
        fig.update_traces(textposition="top left")  # "top right"
        # fig.update_xaxes(range=[94.0, 100.0])
        if smooth_line:
            fig.update_traces(mode="markers+lines", line_shape="spline")

        # remove axis titles
        if template != "plotly_white":
            for axis in fig.layout:
                if type(fig.layout[axis]) == go.layout.YAxis:
                    fig.layout[axis].title.text = ""
                if type(fig.layout[axis]) == go.layout.XAxis:
                    fig.layout[axis].title.text = ""

        if template != "plotly":
            fig.update_layout(
                xaxis=dict(
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    gridcolor="#CECECE",
                ),
                yaxis=dict(
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    gridcolor="#CECECE",
                ),
                xaxis2=dict(
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    gridcolor="#CECECE",
                ),
                yaxis2=dict(
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    gridcolor="#CECECE",
                ),
                template=template,
            )

        if template == "plotly_white":
            yaxis_title = "QPS"
            xaxis_title = "Recall"

        x_legend, y_legend = 0.01, 0.01
        yanchor, xanchor = "bottom", "left"
        if dataset == "LAION-3M" or dataset == "enron":
            # pass
            # move for these plots the legend to the top-right instead of bottom-left
            x_legend, y_legend = 0.99, 0.99
            yanchor, xanchor = "top", "right"

        annotations = None

        fig.update_layout(
            legend=dict(
                xanchor=xanchor,
                yanchor=yanchor,
                x=x_legend,
                y=y_legend,
                # bgcolor="LightGray",
                bordercolor="Black",
                borderwidth=1,
            ),
            font=dict(size=14),
            showlegend=showlegend,
            title_x=0.5,
            autosize=True,
            height=height,
            margin_b=margins[0],
            margin_t=margins[1],
            margin_r=margins[2],
            margin_l=margins[3],
        )

        if template != "plotly_white":
            fig.update_layout(
                annotations=list(fig.layout.annotations)
                + [
                    go.layout.Annotation(
                        x=0.5,
                        y=-0.12,
                        font=dict(size=12),
                        showarrow=False,
                        text=xaxis_title,
                        xref="paper",
                        yref="paper",
                    )
                ]
            )
            fig.add_annotation(
                xref="x domain",
                yref="y domain",
                x=-0.2,
                y=1.05,
                text="QPS",
                showarrow=False,
            )

        fig.update_traces(marker=dict(size=8), line=dict(width=3))

        print("mhs: ", df_plot["mhs"].unique())

        all_figs.append(fig)
        if not suppress_fig_show:
            fig.show()
        if dataset is None:
            return [fig]

    return all_figs


def prepare_df_plot(
    df,
    limit_to_top_layer=False,
    # filters: if empty nothing gets filtered out
    filter_out_hlmhs=[],
    filter_out_k=[],
    filter_out_dataset=[],
    filter_out_builder=[],
):
    pd.to_numeric(df["elapsed_time"])
    pd.to_numeric(df["qps"])
    pd.to_numeric(df["recall"])
    df_plot = pd.DataFrame(
        {
            "builder_type": [],
            "dataset": [],
            "k": [],
            "hlmhs": [],  # higher_level_max_heap_size
            "mhs": [],  # max_heap_size
            "layer": [],
            "qps": [],
            "recall": [],
            "build_time": [],
            "build_time_raw": [],
        }
    )
    if "higher_level_max_heap_size" not in df:
        df["higher_level_max_heap_size"] = ""

    df["dataset"] = df["dataset"].apply(lambda ds: translate_dataset_name(ds))

    # A plot-point for each: builder_type - dataset - k - hlmhs - mhs - layer
    for bt in df["builder_type"].unique():
        if bt in filter_out_builder:
            continue
        df_bt = df.loc[df["builder_type"] == bt]
        for ds in df_bt["dataset"].unique():
            if ds in filter_out_dataset:
                continue
            df_bt_ds = df_bt.loc[df_bt["dataset"] == ds]
            for k in df_bt_ds["k"].unique():
                if k in filter_out_k:
                    continue
                df_bt_ds_k = df_bt_ds.loc[df_bt_ds["k"] == k]
                for hlmhs in df_bt_ds_k["higher_level_max_heap_size"].unique():
                    if hlmhs in filter_out_hlmhs:
                        continue
                    df_bt_ds_k_hlmhs = df_bt_ds_k.loc[
                        df_bt_ds_k["higher_level_max_heap_size"] == hlmhs
                    ]
                    for mhs in df_bt_ds_k["max_heap_size"].unique():
                        df_bt_ds_k_hlmhs_mhs = df_bt_ds_k_hlmhs.loc[
                            df_bt_ds_k["max_heap_size"] == mhs
                        ]
                        layers = df_bt_ds_k_hlmhs_mhs["layer_count"].unique()
                        top_layer = layers.max()
                        for l in df_bt_ds_k_hlmhs_mhs["layer_count"].unique():
                            if limit_to_top_layer and l != top_layer:
                                continue
                            df_bt_ds_k_hlmhs_mhs_layer = df_bt_ds_k_hlmhs_mhs.loc[
                                df_bt_ds_k_hlmhs_mhs["layer_count"] == l
                            ]
                            df_plot.loc[len(df_plot)] = [
                                bt,
                                translate_dataset_name(ds),
                                k,
                                hlmhs,
                                mhs,
                                l,
                                df_bt_ds_k_hlmhs_mhs_layer["qps"].mean(),
                                df_bt_ds_k_hlmhs_mhs_layer["recall"].mean(),
                                f"{int(df_bt_ds_k_hlmhs_mhs_layer['elapsed_time'].iloc[0])}s",
                                df_bt_ds_k_hlmhs_mhs_layer["elapsed_time"].iloc[0],
                            ]
    return df_plot


def plot_build_times(df, template="plotly"):
    # backward-compatibility fix
    df = df.rename(columns={"data_set": "dataset"})
    df_plot = prepare_df_plot(df)
    datasets = df_plot["dataset"].unique()
    fig = make_subplots(
        rows=1, cols=len(datasets), shared_yaxes=False, subplot_titles=datasets
    )
    for i, ds in enumerate(datasets):
        df_sub = df_plot.loc[df_plot["dataset"] == ds]
        fig.add_trace(
            go.Bar(
                x=df_sub["builder_type"].unique(),
                y=df_sub["build_time_raw"].unique(),
                # marker=dict(color=list(range(0, len(datasets)))),
                marker_color=[
                    # use the classical plotly.express colors as for the other plots
                    "#636EFA",
                    "#EF553B",
                    "#00CC96",
                    "#AB63FA",
                    "#FFA15A",
                ],
                showlegend=False,
                name="",
            ),
            1,
            i + 1,
        )
        if template == "plotly_white":
            fig.update_xaxes(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                row=1,
                col=i + 1,
            )
            fig.update_yaxes(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                row=1,
                col=i + 1,
            )

    fig.update_yaxes(title_text="Build time in seconds", row=1, col=1)

    margins = [45, 20, 20, 50]
    fig.update_layout(
        autosize=True,
        font=dict(size=16),
        margin_b=margins[0],
        margin_t=margins[1],
        margin_r=margins[2],
        margin_l=margins[3],
        template=template,
        showlegend=False,
    )

    fig.show()
    return fig


def plot_hierarchy_graphs_compare(df, k, hlmhs=1, only_show_bottom_top_points=False):
    df_plot = prepare_df_plot(df)
    df_plot = df_plot.loc[df_plot["hlmhs"] == hlmhs]
    df_plot = df_plot.loc[df_plot["k"] == k]

    if only_show_bottom_top_points:
        df_plot = df_plot.loc[
            (
                (df_plot["layer"] == 1)
                # uncomment, among others, to compare different hnsw graphs
                # between each other
                # | ((df_plot["builder_type"] != "HNSW") & (df_plot["layer"] == 4))
                | (
                    (
                        df_plot["builder_type"]
                        != "HSGF_DEG_Flood_2_Flood_1_lower_degree_higher"
                    )
                    & (df_plot["layer"] == 3)
                )
                | (df_plot["layer"] == 5)
            )
        ]

    fig = px.scatter(
        df_plot,
        title=f"{list(df_plot['dataset'].unique())[0]}",
        x="recall",
        y="qps",
        # text="layer",
        hover_data=[
            "layer",
            "mhs",
            "builder_type",
            "dataset",
            "build_time",
            "hlmhs",
        ],
        color="builder_type",
    )

    # display(df_plot.head(1))
    # display(df_plot.tail(1))

    colors = [["blue"], ["red"], ["#17cfa7"]]
    for i, bt in enumerate(df_plot["builder_type"].unique()):
        df_local = df_plot.loc[df_plot["builder_type"] == bt]
        df_local_ = df_local.loc[df_local["layer"] == int(1)]
        fig.add_traces(
            list(
                px.line(
                    df_local_,
                    x="recall",
                    y="qps",
                    line_shape="spline",
                    color_discrete_sequence=colors[i],
                ).select_traces()
            )
        )
        top_layer = 3  # change manually
        if bt == "HNSW" or bt == "HSGF_DEG_Flood_2_Flood_1_lower_degree_higher":
            top_layer = 5
        df_local_ = df_local.loc[df_local["layer"] == int(top_layer)]
        fig.add_traces(
            list(
                px.line(
                    df_local_,
                    x="recall",
                    y="qps",
                    line_shape="spline",
                    color_discrete_sequence=colors[i],
                ).select_traces()
            )
        )

    margins = [60, 40, 0, 0]
    fig.update_layout(
        autosize=True,
        title_x=0.5,
        xaxis_title="Recall",
        yaxis_title="QPS",
        template="plotly_white",
        font=dict(size=14),
        legend=dict(
            yanchor="bottom",
            xanchor="left",
            x=0.01,
            y=0.01,
            bordercolor="Black",
            borderwidth=1,
        ),
        margin_b=margins[0],
        margin_t=margins[1],
        margin_r=margins[2],
        margin_l=margins[3],
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            gridcolor="#CECECE",
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            gridcolor="#CECECE",
        ),
    )
    fig.update_traces(textposition="top left")

    fig.show()
    return fig


def plot_hierarchy_ef_higher_compare(df, k, only_show_bottom_top_points=False):
    df_plot = prepare_df_plot(df)
    # df_plot = df_plot.loc[df_plot["hlmhs"] == 1]
    df_plot = df_plot.loc[df_plot["k"] == k]

    layers = df_plot["layer"].unique()
    top_layer = layers.max()

    if only_show_bottom_top_points:
        df_plot = df_plot.loc[
            ((df_plot["layer"] == 1) | (df_plot["layer"] == top_layer))
        ]

    color_by = "hlmhs"
    df_plot[color_by] = df_plot[color_by].astype(str)

    fig = px.scatter(
        df_plot,
        title=f"{list(df_plot['dataset'].unique())[0]}",
        x="recall",
        y="qps",
        hover_data=[
            "layer",
            "mhs",
            "builder_type",
            "dataset",
            "build_time",
            "hlmhs",
        ],
        color=color_by,
        labels={
            "builder_type": "Graph-Type",
            "mhs": "$ef_{bottom}$",
            "hlmhs": "$ef_{higher}$",
            "qps": "QPS",
            "recall": "Recall",
        },
    )
    only_once = True

    colors = [["#636EFA"], ["#EF553B"], ["#00CC96"], ["#AB63FA"], ["#FFA15A"]]
    for i, hlmhs in enumerate(df_plot["hlmhs"].unique()):
        df_local = df_plot.loc[df_plot["hlmhs"] == hlmhs]
        if only_once:
            only_once = False
            df_local_ = df_local.loc[df_local["layer"] == int(1)]
            fig.add_traces(
                list(
                    px.line(
                        df_local_,
                        x="recall",
                        y="qps",
                        line_shape="spline",
                        color_discrete_sequence=colors[i],
                    ).select_traces()
                )
            )
        df_local_ = df_local.loc[df_local["layer"] == top_layer]
        fig.add_traces(
            list(
                px.line(
                    df_local_,
                    x="recall",
                    y="qps",
                    line_shape="spline",
                    color_discrete_sequence=colors[i],
                ).select_traces()
            )
        )

    # remove axis titles
    # for axis in fig.layout:
    #     if type(fig.layout[axis]) == go.layout.YAxis:
    #         fig.layout[axis].title.text = ""
    #     if type(fig.layout[axis]) == go.layout.XAxis:
    #         fig.layout[axis].title.text = ""

    margins = [50, 35, 10, 0]
    ks = "".join([f"{e}," for e in df_plot["k"].unique()])[:-1]
    mhss = "".join([f"{e}," for e in df_plot["mhs"].unique()])[:-1]
    margins = [60, 40, 0, 0]
    fig.update_layout(
        autosize=True,
        title_x=0.5,
        # xaxis_title=f"k={ks}@ef=[{mhss}]-Recall",
        xaxis_title="Recall",
        yaxis_title="QPS",
        template="plotly_white",
        legend=dict(
            yanchor="bottom",
            xanchor="left",
            x=0.01,
            y=0.01,
            bordercolor="Black",
            borderwidth=1,
        ),
        margin_b=margins[0],
        margin_t=margins[1],
        margin_r=margins[2],
        margin_l=margins[3],
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            gridcolor="#CECECE",
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            gridcolor="#CECECE",
        ),
    )
    print("mhs: ", df_plot["mhs"].unique())

    fig.show()
    return fig


# --------- Helper ---------


def translate_dataset_name(dataset):
    if dataset == "300K" or dataset == "3M" or dataset == "10M" or dataset == "100M":
        dataset = f"LAION-{dataset}"
    return dataset


# --------- Create Compare Gains Table Helpers ---------


def create_gains_table(df, filename_res=None, k=1, hlmhs=1):
    bottom, top_4, top_5, top_6 = (1, 4, 5, 6)
    # just pre-filter dataframe
    df = df.loc[
        (
            (df["layer"] == bottom)
            | (df["layer"] == top_4)
            | (df["layer"] == top_5)
            | (df["layer"] == top_6)
        )
    ]

    print(df["builder_type"].unique())
    pd.to_numeric(df["k"])
    pd.to_numeric(df["hlmhs"])
    df = df.loc[((df["k"] == k) & (df["hlmhs"] == hlmhs))]

    df_res = pd.DataFrame(
        {
            "builder_type": [],
            "k": [],
            "hlmhs": [],
            "mhs": [],
            "qps_diff_percentage": [],
            "qps_diff": [],
            "recall_diff_percentage": [],
            "recall_diff": [],
            "recall_bottom": [],
            "qps_bottom": [],
        }
    )

    for i, bt in enumerate(df["builder_type"].unique()):
        df_bt = df.loc[df["builder_type"] == bt]
        for k__ in df_bt["k"].unique():
            df_local = df_bt.loc[df_bt["k"] == k__]
            for hlmhs__ in df_local["hlmhs"].unique():
                df_local_hlmhs = df_local.loc[df_local["hlmhs"] == hlmhs__]
                for mhs in df_local_hlmhs["mhs"].unique():
                    df_x = df_local_hlmhs.loc[df_local_hlmhs["mhs"] == mhs]
                    layers = df_x["layer"].unique()
                    top_layer = layers.max()
                    l_bottom = df_x.loc[df_x["layer"] == bottom]
                    l_top = df_x.loc[df_x["layer"] == top_layer]
                    qps_diff = float(l_top["qps"].iloc[0]) - float(
                        l_bottom["qps"].iloc[0]
                    )
                    recall_diff = float(l_top["recall"].iloc[0]) - float(
                        l_bottom["recall"].iloc[0]
                    )
                    df_res.loc[len(df_res)] = [
                        bt,
                        int(k__),
                        int(hlmhs__),
                        int(mhs),
                        (qps_diff / float(l_bottom["qps"].iloc[0])) * 100,
                        qps_diff,
                        (recall_diff / float(l_bottom["recall"].iloc[0])) * 100,
                        recall_diff,
                        l_bottom["recall"].iloc[0],
                        l_bottom["qps"].iloc[0],
                    ]

    if filename_res != None:
        df_res.to_csv(
            filename_res,
            sep=";",
            index=False,
            mode="a",
        )

    # ----- Print in Latex table format -----

    # might need manual filtering here
    # df_res = df_res.loc[df_res["builder_type"] == "HNSW"]
    df_res = df_res.loc[((df_res["k"] == k) & (df_res["hlmhs"] == hlmhs))]
    print(df_res["builder_type"].unique())

    res_str = ""
    for mhs in df_res["mhs"].unique():
        # if int(mhs) > 200 or int(mhs) == 150:
        #     continue
        df_mhs = df_res.loc[df_res["mhs"] == mhs]
        res_str += f"{mhs}"
        # for bt in [""]:
        for bt in df_res["builder_type"].unique():
            df_mhs_bt = df_mhs.loc[df_mhs["builder_type"] == bt]
            res_str += f' & +{round(df_mhs_bt["recall_diff_percentage"].iloc[0], 2)}\\% & {round(df_mhs_bt["recall_bottom"].iloc[0], 2)}\\% & +{round(df_mhs_bt["qps_diff_percentage"].iloc[0], 2)}\\% & {round(df_mhs_bt["qps_bottom"].iloc[0], 2)}'
        res_str += r" \\ \hline" + "\n"

    print(res_str)
    return res_str
