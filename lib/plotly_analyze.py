import numpy as np
import plotly as py
import chart_studio
import plotly.io as pio
import plotly.graph_objects as go
from plotly.graph_objs import *
import torch
import math
import pandas as pd
import plotly.express as px
import os
from pandas.core.frame import DataFrame

pio.templates.default = "simple_white"
pyplt = py.offline.plot
p = pio.renderers['png']
p.width = 800
p.height = 600
chart_studio.tools.set_config_file(world_readable=True, sharing='public')


def plotly_print_grad(path):
    m = np.array(pd.read_csv(os.path.join(path, str(0) + '.csv'), sep=',', index_col=False,
                             encoding="utf-8", low_memory=False))
    xx = np.linspace(0, m.shape[0] - 1, m.shape[0])
    yy = np.linspace(0, m.shape[1] - 1, m.shape[1])
    # C={"z":{"show":True,"start":-1,"end":1,"size":0.2,"usecolormap":True,"project_z":True,"highlightcolor":"limegreen"}}
    fig = go.Figure(
        data=[go.Surface(x=xx, y=yy, z=m, colorscale='Viridis')],
        frames=[]
    )
    for i in range(1, 700, 1):
        ture_path = os.path.join(path, str(i) + '.csv')
        m = np.array(pd.read_csv(ture_path, sep=',', index_col=False,
                                 encoding="utf-8", low_memory=False))

        fig.frames += (go.Frame(data=[go.Surface(x=xx, y=yy, z=m)]),)
    # data=DataFrame(data,columns=['times','row','col','value'])
    print(fig.frames)
    fig.update_layout(
        title_text='different function of x and y',
        height=800,
        width=800,
        scene=py.graph_objs.layout.Scene(
            xaxis=py.graph_objs.layout.scene.XAxis(
                range=[0, m.shape[0]],
                nticks=10,
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=py.graph_objs.layout.scene.YAxis(
                range=[0, m.shape[1]],
                nticks=10,
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
        ),
        legend=dict(
            x=0.1,
            y=0.9,
            bgcolor="white",
            bordercolor="black",
            borderwidth=2,
        )
        , updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    )
    fig.show()


if __name__ == "__main__":
    plotly_print_grad('../output/error/tau_m2')
