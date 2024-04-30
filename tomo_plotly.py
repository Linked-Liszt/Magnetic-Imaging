import dash
from dash import dcc, ctx
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import tomopy
from dataclasses import dataclass

# Assume images are numpy arrays
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
# Create a Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc_css], suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


base = '../data_base/staged/'
images_1_l = np.load(base + '0002_tomogram1-stxm_cl-150-cor.npy')
images_1_r = np.load(base + '0002_tomogram1-stxm_cr-150-cor.npy')

images_2_l = np.load(base + '0004_tomogram2-stxm_cl-50-cor.npy')
images_2_r = np.load(base + '0004_tomogram2-stxm_cr-50-cor.npy')

images_3_l = np.load(base + '0007_tomogram3-stxm_cl-135.npy')
images_3_r = np.load(base + '0007_tomogram3-stxm_cr-135.npy')

images_4_l = np.load(base + '0008_tomogram4-stxm_cl-178.npy')
images_4_r = np.load(base + '0008_tomogram4-stxm_cr-178.npy')

@dataclass
class ReconState:
    norm: bool
    scale: bool
    scale_factor: float

STATE = {
    'curr_im':'load_1',
    'states':{
        'load_1': ReconState(False, False, 0.9),
        'load_2': ReconState(False, False, 0.9),
        'load_3': ReconState(False, False, 0.9),
        'load_4': ReconState(False, False, 0.9),
    }
}


"""
=================================
UI Components
=================================
"""

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Data Import", href="/", active=True)),
        dbc.NavItem(dbc.NavLink("Reconstruction", href="/reconstruction")),
        dbc.NavItem(dbc.NavLink("Visualization", href="#")),

    ],
    brand="Magnetic Tomo Reconstruction",
    brand_href="/",
    color="primary",
    className="navbar-expand-lg",
    dark=True,
    style={"max-height": "50px"},
)

accordian = [
    dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    dbc.Input(id='tomo_1', value='0002_tomogram1-stxm_cl-150-cor.npy'),
                    dbc.Button("Load", id='demo_1', disabled=True),
                    dbc.Button("Display", id='load_1'),
                ],
                title="Projections 1",
            ),
            dbc.AccordionItem(
                [
                    dbc.Input(id='tomo_2', value='0004_tomogram2-stxm_cl-50-cor.npy'),
                    dbc.Button("Load", id='demo_2', disabled=True),
                    dbc.Button("Display", id='load_2'),
                ],
                title="Projections 2",
            ),
            dbc.AccordionItem(
                [
                    dbc.Input(id='tomo_3', value='0007_tomogram3-stxm_cl-135.npy'),
                    dbc.Button("Load", id='demo_3', disabled=True),
                    dbc.Button("Display", id='load_3'),
                ],
                title="Projections 3",
            ),
            dbc.AccordionItem(
                [
                    dbc.Input(id='tomo_4', value='0008_tomogram4-stxm_cl-178.npy'),
                    dbc.Button("Load", id='demo_4', disabled=True),
                    dbc.Button("Display", id='load_4'),
                ],
                title="Projections 4",
            ),
        ],
    )
]

display_control = [
    html.Div([
        dcc.Graph(id='image', style={'width': '50%'}),
        dcc.Graph(id='processed-image', style={'width': '50%'})
    ], style={'display': 'flex'}),
    dbc.Label("Select an image to display:", html_for='image-slider'),
    dcc.Slider(
        id='image-slider',
        min=0,
        max=len(images_1_l) - 1,
        value=0,
        step=1,
        marks={i: str(i) for i in range(0, len(images_1_l), 50)},
    ),
    dbc.Checkbox(id='scale', label='Scaling'),
    dbc.Input(id='scale_factor', type='number', placeholder='Scaling Factor', value=0.9),
    dbc.Checkbox(id='clip', label='Clipping'),
    dbc.Checkbox(id='norm', label='Tomopy Normalization'),
    dbc.Button('Excluded Image'),
    dbc.Input(value='TODO: Excluded images here')
]

# Add a Plotly graph and a slider to the application layout
load_layout = dbc.Container(
    [html.Div([
        navbar,
    dbc.Row(
        [
            dbc.Col(accordian, width=3),
            dbc.Col(display_control, width=9),
        ]
    )
    ],
    )
    ],
    className='dbc', 
    fluid=True
)


recon_layout = dbc.Container(
    [html.Div([
        navbar,
        html.H1("Reconstruction"),
        dbc.Button("Align (TODO)", id='align'),
        dcc.Graph(id='recon', style={'width': '50%'}),
    ],
    )
    ],
    className='dbc', 
    fluid=True
)

"""
=================================
Callbacks
=================================
"""

# Define a callback that changes the image displayed in the graph when the slider value changes
@app.callback(
    [Output('image', 'figure'), Output('processed-image', 'figure'), Output('image-slider', 'max'), Output('image-slider', 'marks')],
    [Input('image-slider', 'value'), Input('norm', 'value'), Input('scale', 'value'), Input('scale_factor', 'value'), Input('clip', 'value'),
     Input('load_1', 'n_clicks'), Input('load_2', 'n_clicks'), Input('load_3', 'n_clicks'), Input('load_4', 'n_clicks')]
)
def update_image(slider_value, norm, scale, scale_factor, clip,
                 load_1, load_2, load_3, load_4):
    global STATE

    if ctx.triggered_id in STATE['states'].keys():
        STATE['curr_im'] = ctx.triggered_id

    if "load_1" == STATE['curr_im']:
        images = np.concatenate((images_1_l, images_1_r), axis=2)
    elif "load_2" == STATE['curr_im']:
        images = np.concatenate((images_2_l, images_2_r), axis=2)
    elif "load_3" == STATE['curr_im']:
        images = np.concatenate((images_3_l, images_3_r), axis=2)
    elif "load_4" == STATE['curr_im']:
        images = np.concatenate((images_4_l, images_4_r), axis=2)

    STATE['states'][STATE['curr_im']].norm = norm
    STATE['states'][STATE['curr_im']].norm = scale
    STATE['states'][STATE['curr_im']].norm = scale_factor

    processed_image = np.copy(images[slider_value])

    if scale:
        processed_image = processed_image / scale_factor

    if clip:
        processed_image[processed_image > 1] = 0.995
        processed_image[processed_image > 1] = 0.995


    if norm: 
        processed_image = np.expand_dims(processed_image, axis=0)
        processed_image = tomopy.normalize_bg(processed_image)
        processed_image = tomopy.minus_log(processed_image)
        processed_image = np.squeeze(processed_image, axis=0)
        processed_image = processed_image / scale_factor
    
    # Display the original image
    original_figure = px.imshow(images[slider_value], color_continuous_scale='gray', title='Original Image')
    
    # Display the processed image
    processed_figure = px.imshow(processed_image, color_continuous_scale='gray', title='Preprocessed Image')


    max_slider = len(images) - 1
    marks = {i: str(i) for i in range(0, len(images), 50)}
    marks[len(images) - 1] = str(len(images) - 1)
    
    return original_figure, processed_figure, max_slider, marks


@app.callback(
    [Output('recon', 'figure')],
    [Input('align', 'n_clicks')],
    prevent_initial_call=True,
    running=[(Output("align", "disabled"), True, False)]
)
def do_recon(n_clicks):
    global STATE
    if n_clicks is None:
        # Button has not been clicked yet, don't do anything
        return dash.no_update

    prj = np.r_[images_1_l, images_1_r]
    ang = np.load(base + '0002_tomogram1-ang-150.npy')
    ang = np.r_[ang, ang]
    print('started')
    # TOOD launch subprocess
    # Try recon, try not sirt
    prj, sx, sy, conv = tomopy.align_joint(
            prj, ang, fdir='aligned/0002_tomogram1/', iters=1, pad=(0, 0),
            blur=True, center=None, algorithm='sirt',
            upsample_factor=100, rin=0.5, rout=0.8,
            save=False, debug=True)
    print('Finished')
    processed_figure = px.imshow(prj[0], color_continuous_scale='gray', title='Preprocessed Image')
    return processed_figure

# Update the index
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return load_layout
    elif pathname == '/reconstruction':
        return recon_layout

    #else:
    #    return index_page
    # You could also return a 404 "URL not found" page here


# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)