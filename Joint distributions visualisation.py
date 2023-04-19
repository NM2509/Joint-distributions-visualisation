#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anastasiamalakhova
"""

import numpy as np 
import math
from scipy.special import gamma
from scipy.stats import t, norm
import plotly.graph_objects as go
import plotly.io as pio

import dash
from dash import dcc, html 
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output 
import webbrowser
from threading import Timer


# Initial graph
n = 100
X = np.linspace(0,5,n)
Y = np.linspace(0,5,n)
lambda_1 = 1
lambda_2 = 1

def joint_pdf_exponential(x,y):
    f_xy = lambda_1*math.exp(-lambda_1*x)*lambda_2*math.exp(-lambda_2*y)
    return f_xy

Z = []

for x in X:
    for y in Y:
        entry = joint_pdf_exponential(x,y)
        Z.append(entry)
Z = np.asarray(Z)

X = np.repeat(list(X),n)
Y = np.array(list(Y)*n)


# Initialising figures - DASH

app = dash.Dash(__name__,  external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

colors = {'background': 'rgb(223,245,249)'}

fig2 = go.Figure(go.Mesh3d(x=X,y=Y,
                          z=np.array(Z),opacity=0.7, 
                          colorscale="greens", intensity=np.array(Z), 
                          visible = True))

fig2.update_layout(paper_bgcolor = colors['background'], 
                  plot_bgcolor = colors['background'], 
                  width=800,
                  height=700,
                  autosize=False,
                  margin=dict(t=70, b=70, l=0, r=0),
                  scene = dict(
                    xaxis_title='Exponential({})'.format(lambda_1),
                    yaxis_title= 'Exponential({})'.format(lambda_2),
                    zaxis_title='Joint pdf'),
                  )

# App layout

app.layout = html.Div(children=[
    
    html.H2(children='A tool for visualising joint distributions', 
            style={'backgroundColor':colors['background'], 'text-align':'center', 
                   'display':'grid', 'grid-column-start': '1','grid-column-end': '3', 'grid-row-start': '1'}),

    # First distribution
    html.Div(children='''
             Please choose the first distribution: 
             ''', style={'backgroundColor':colors['background'], 'margin-top': '3vw','font-weight':'bold', 'display':'grid', 'grid-column-start': '1'}),
             
    dcc.Dropdown(id = 'dropdown_first_graph', 
    options=['normal', 'exponential', 'gamma', 't-distribution'],
    value='exponential',
    style={'margin-top': '0vw', 'margin-bottom':'1vw', 'display':'grid',   'grid-column-start': '1','grid-column-end': '1', 'width':'300px'}),
    
    
    html.Div(id = 'name1', children = '''Mean''', style={'backgroundColor':colors['background'], 
                ' margin-top': '2vw',  'margin-left': '0px', 'display': 'grid',  'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id="mu1_", type="text", value = 10), ],  style={ 'margin-top': '0vw', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}),
    
    html.Div(id = 'name2', children = '''Standard deviation''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '1vw',  'margin-left': '0px', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id='std1_', type="text", value = 5), ],  style={ 'margin-top': '0vw', 'display': 'grid','grid-column-start': '1', 'width':'100px'}),
    
    html.Div(id = 'name3', children = '''Rate - lambda''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '2vw',  'margin-left': '0px', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id='lambda1_', type="text", value = 1), ],  style={'display': 'grid', 'grid-column-start': '1', 'width':'100px'}),
    
    html.Div(id = 'name7', children = '''Shape (alpha)''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '2vw',  'margin-left': '0px', 'display': 'grid',  'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id="shape1", type="text", value = 2), ],  style={'margin-top': '0vw', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div(id = 'name8', children = '''Rate (beta)''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '1vw',  'margin-left': '0px', 'display': 'grid',  'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id='rate1', type="text", value = 0.5), ],  style={ 'margin-top': '0vw', 'display': 'grid',  'grid-column-start': '1', 'width':'100px'}),
    
    
    html.Div(id = 'name11', children = '''Degrees of freedom''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '1vw',  'margin-left': '0px', 'display': 'grid',  'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id='dof1', type="text", value = 10), ],  style={ 'margin-top': '0vw', 'display': 'grid',  'grid-column-start': '1', 'width':'100px'}),
    
    
    # Second distribution
    html.Div(children='''
             Please choose the second distribution: 
             ''', style={'backgroundColor':colors['background'], 'margin-top': '2vw', 'font-weight':'bold', 'display':'grid',  'grid-column-start': '1'}),

    dcc.Dropdown(id = 'dropdown_second_graph', 
    options=['normal', 'exponential', 'gamma', 't-distribution'],
    value='exponential',
    style={'margin-top': '0vw','margin-bottom':'1vw', 'display':'grid',  'grid-column-start': '1', 'width':'300px'}),
    
     html.Div(id = 'name4', children = '''Mean''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '2vw',  'margin-left': '0px', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id="mu2_", type="text", value = 10), ],  style={ 'margin-top': '0vw', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}),
    
    html.Div(id = 'name5', children = '''Standard deviation''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '0vw',  'margin-left': '0px', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id='std2_', type="text", value = 5), ],  style={ 'margin-top': '0vw', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}),
    
    html.Div(id = 'name6', children = '''Rate - lambda''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '2vw',  'margin-left': '0px', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id='lambda2_', type="text", value = 1), ],  style={ 'margin-top': '0vw', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}),
    
    html.Div(id = 'name9', children = '''Shape (alpha)''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '2vw',  'margin-left': '0px', 'display': 'grid',  'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id="shape2", type="text", value = 2), ],  style={ 'margin-top': '0vw', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}),
    
    html.Div(id = 'name10', children = '''Rate (beta)''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '1vw',  'margin-left': '0px', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id='rate2', type="text", value = 0.5), ],  style={ 'margin-top': '0vw', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}),
    
    
    html.Div(id = 'name12', children = '''Degrees of freedom''', style={'backgroundColor':colors['background'], 
                 ' margin-top': '1vw',  'margin-left': '0px', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}), 
    
    html.Div([
        dcc.Input(id='dof2', type="text", value = 10), ],  style={ 'margin-top': '0vw', 'display': 'grid', 'grid-column-start': '1', 'width':'100px'}),
    
    # Graph
    html.Div(children = [ 
        dcc.Graph(
        id='graph2',
        figure=fig2,)],
        style={'margin-left': '3vw', 'margin-top': '1vw', 'margin-right': '5vw', 'display': 'grid',  'grid-column-start': '2','grid-row-start': '2', 'grid-row-end': '40'}),
      
    ],
             
    # Style of the layout    
    style={'backgroundColor':colors['background'], 'display': 'grid', 'grid-template-columns': '30% 70%'})
             

             
# Template - background of the graph
pio.templates["custom_dark"] = go.layout.Template(
    layout=go.Layout(
        colorway=['#ff0000', '#00ff00', '#0000ff']))
      

pio.templates['custom_dark']['layout']['yaxis']['gridcolor'] = 'rgb(223,245,249)'
pio.templates['custom_dark']['layout']['xaxis']['gridcolor'] = 'rgb(223,245,249)'
fig2.layout.template = 'custom_dark'

# Callbacks

# Parameters

# First distribution - mean name
@app.callback(
   Output(component_id='name1', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element0(visibility_state):
    if visibility_state == 'normal':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}

# First distribution - mean
@app.callback(
   Output(component_id='mu1_', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element1(visibility_state):
    if visibility_state == 'normal':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
# First distribution - standard deviation name
@app.callback(
   Output(component_id='name2', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element2(visibility_state):
    if visibility_state == 'normal':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
    
# First distribution - standard deviation 
@app.callback(
   Output(component_id='std1_', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element3(visibility_state):
    if visibility_state == 'normal':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
    
# First distribution - exponential name 
@app.callback(
   Output(component_id='name3', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element4(visibility_state):
    if visibility_state == 'exponential':
        return {'display': 'block'} 
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    

# First distribution - exponential
@app.callback(
   Output(component_id='lambda1_', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element5(visibility_state):
    if visibility_state == 'exponential':
        return {'display': 'block'} 
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
# First distribution - shape name
@app.callback(
   Output(component_id='name7', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element00(visibility_state):
    if visibility_state == 'gamma':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}

# First distribution - shape
@app.callback(
   Output(component_id='shape1', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element10(visibility_state):
    if visibility_state == 'gamma':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
# First distribution - rate name
@app.callback(
   Output(component_id='name8', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element2_(visibility_state):
    if visibility_state == 'gamma':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
    
# First distribution - rate
@app.callback(
   Output(component_id='rate1', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element30(visibility_state):
    if visibility_state == 'gamma':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
    
# First distribution - degrees of freedom name
@app.callback(
   Output(component_id='name11', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element000(visibility_state):
    if visibility_state == 't-distribution':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}

# First distribution - degrees of freedom
@app.callback(
   Output(component_id='dof1', component_property='style'),
   [Input(component_id='dropdown_first_graph', component_property='value')])


def show_hide_element100(visibility_state):
    if visibility_state == 't-distribution':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    


# Second distribution - mean name
@app.callback(
   Output(component_id='name4', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element6(visibility_state):
    if visibility_state == 'normal':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
    
# Second distribution - mean
@app.callback(
   Output(component_id='mu2_', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element7(visibility_state):
    if visibility_state == 'normal':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
   
    
# Second distribution - standard deviation name
@app.callback(
   Output(component_id='name5', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element8(visibility_state):
    if visibility_state == 'normal':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
    
# Second distribution - standard deviation 
@app.callback(
   Output(component_id='std2_', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element9(visibility_state):
    if visibility_state == 'normal':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
# First distribution - exponential name 
@app.callback(
   Output(component_id='name6', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element1000(visibility_state):
    if visibility_state == 'exponential':
        return {'display': 'block'} 
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
# Second distribution - exponential
@app.callback(
   Output(component_id='lambda2_', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element11(visibility_state):
    if visibility_state == 'exponential':
        return {'display': 'block'} 
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
# Second distribution - shape name
@app.callback(
   Output(component_id='name9', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element60(visibility_state):
    if visibility_state == 'gamma':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
    
# Second distribution - shape
@app.callback(
   Output(component_id='shape2', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element70(visibility_state):
    if visibility_state == 'gamma':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
   
    
# Second distribution - rate name
@app.callback(
   Output(component_id='name10', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element80(visibility_state):
    if visibility_state == 'gamma':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    
    
# Second distribution - rate
@app.callback(
   Output(component_id='rate2', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element90(visibility_state):
    if visibility_state == 'gamma':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    if visibility_state == 't-distribution':
        return {'display': 'none'}
    

# Second distribution - degrees of freedom name
@app.callback(
   Output(component_id='name12', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element0000(visibility_state):
    if visibility_state == 't-distribution':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma': 
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    

# Second distribution - degrees of freedom
@app.callback(
   Output(component_id='dof2', component_property='style'),
   [Input(component_id='dropdown_second_graph', component_property='value')])


def show_hide_element10000(visibility_state):
    if visibility_state == 't-distribution':
        return {'display': 'block'} 
    if visibility_state == 'exponential':
        return {'display': 'none'}
    if visibility_state == 'gamma':
        return {'display': 'none'}
    if visibility_state == 'normal':
        return {'display': 'none'}
    
    
@app.callback(
    Output('graph2', 'figure'),
    Input('dropdown_first_graph', 'value'),
    Input('dropdown_second_graph', 'value'),
    Input('lambda1_', 'value'),
    Input('lambda2_', 'value'),
    Input('mu1_', 'value'),
    Input('std1_', 'value'),
    Input('mu2_', 'value'),
    Input('std2_', 'value'), 
    Input('shape1', 'value'),
    Input('rate1', 'value'),
    Input('shape2', 'value'),
    Input('rate2', 'value'), 
    Input('dof1', 'value'), 
    Input('dof2', 'value'), 
    )

def update_output2(dropdown, dropdown2, lambda1_value, lambda2_value, mu1_value, std1_value, mu2_value, std2_value, shape1, rate1, shape2, rate2, dof1, dof2):

    def joint_pdf_exponential_exponential(x,y):
        f_xy = lambda_1*math.exp(-lambda_1*x)*lambda_2*math.exp(-lambda_2*y)
        return f_xy
    
    def joint_pdf_exponential_normal(x,y):
        f_xy = lambda_1*math.exp(-lambda_1*x)*(1/(std_2*math.sqrt(2*math.pi)))*math.exp(-0.5*(((y-mu_2)/std_2)**2))
        return f_xy
    
    def joint_pdf_normal_exponential(x,y):
        f_xy = (1/(std_1*math.sqrt(2*math.pi)))*math.exp(-0.5*(((x-mu_1)/std_1)**2))*lambda_2*math.exp(-lambda_2*y)
        return f_xy
    
    def joint_pdf_exponential_gamma(x,y):
        f_xy = lambda_1*math.exp(-lambda_1*x)*((rate_2**shape_2)*(y**(shape_2-1))*math.exp(-rate_2*y))/math.factorial(shape_2-1)
        return f_xy
    
    def joint_pdf_gamma_exponential(x,y):
        f_xy = (((rate_1**shape_1)*(x**(shape_1-1))*math.exp(-rate_1*x))/math.factorial(shape_1-1))*(lambda_2*math.exp(-lambda_2*y))
        return f_xy
    
    def joint_pdf_gamma_gamma(x,y):
        f_xy = (((rate_1**shape_1)*(x**(shape_1-1))*math.exp(-rate_1*x))/math.factorial(shape_1-1))*((rate_2**shape_2)*(y**(shape_2-1))*math.exp(-rate_2*y))/math.factorial(shape_2-1)
        return f_xy
    
    def joint_pdf_gamma_normal(x,y):
        f_xy = (((rate_1**shape_1)*(x**(shape_1-1))*math.exp(-rate_1*x))/math.factorial(shape_1-1))*(1/(std_2*math.sqrt(2*math.pi)))*math.exp(-0.5*(((y-mu_2)/std_2)**2))
        return f_xy
    
    def joint_pdf_normal_gamma(x,y):
        f_xy = ((1/(std_1*math.sqrt(2*math.pi)))*math.exp(-0.5*(((x-mu_1)/std_1)**2)))*(((rate_2**shape_2)*(y**(shape_2-1))*math.exp(-rate_2*y))/math.factorial(shape_2-1))
        return f_xy
        
    def joint_pdf_t_dist_gamma(x,y):
        f_xy = (gamma((dof_1+1)/2)*(1+((x**2)/dof_1))**(-0.5*(dof_1+1)))/(math.sqrt(math.pi*dof_1)*gamma(dof_1/2))*((rate_2**shape_2)*(y**(shape_2-1))*math.exp(-rate_2*y))/math.factorial(shape_2-1)
        return f_xy
    
    def joint_pdf_t_dist_exponential(x,y):
        f_xy = (gamma((dof_1+1)/2)*(1+((x**2)/dof_1))**(-0.5*(dof_1+1)))/(math.sqrt(math.pi*dof_1)*gamma(dof_1/2))*(lambda_2*math.exp(-lambda_2*y))
        return f_xy
                    
    def joint_pdf_gamma_t_dist(x,y):
        f_xy = (((rate_1**shape_1)*(x**(shape_1-1))*math.exp(-rate_1*x))/math.factorial(shape_1-1))*(gamma((dof_2+1)/2)*(1+((y**2)/dof_2))**(-0.5*(dof_2+1)))/(math.sqrt(math.pi*dof_2)*gamma(dof_2/2))
        return f_xy
        
    def joint_pdf_exponential_t_dist(x,y):
        f_xy = lambda_1*math.exp(-lambda_1*x)*(gamma((dof_2+1)/2)*(1+((y**2)/dof_2))**(-0.5*(dof_2+1)))/(math.sqrt(math.pi*dof_2)*gamma(dof_2/2))
        return f_xy
            
    
    # Normal 1   
    if mu1_value == '':
        mu_1 = 10
    else:
        mu_1 = float(mu1_value)     
        
    if std1_value == '':
        std_1 = 3
    else:
        std_1 = float(std1_value)
        
    # Normal 2   
    if mu2_value == '':
        mu_2 = 10
    else:
        mu_2 = float(mu2_value)     
        
    if std2_value == '':
        std_2 = 3
    else:
        std_2 = float(std2_value)
        
    # Exponential 1 
    if lambda1_value == '':
        lambda_1 = 1
    else:
        lambda_1 = float(lambda1_value)
    
    # Exponential 2
    if lambda2_value == '':
        lambda_2 = 1
    else: 
        lambda_2 = float(lambda2_value)

    # Gamma 1 
    if shape1 == '':
        shape_1 = 2
    else: 
        shape_1 = float(shape1)
    
    if rate1 == '':
        rate_1 = 0.5
    else: 
        rate_1 = float(rate1)
        
    # Gamma 2     
    if shape2 == '':
        shape_2 = 2
    else: 
        shape_2 = float(shape2)
          
    if rate2 == '':
        rate_2 = 0.5
    else: 
        rate_2 = float(rate2)
        
    # t-distribution 1 
    if dof1 == '':
        dof_1 = 10
    else:
        dof_1 = float(dof1)
    
    # t-distribution 2
    if dof2 == '':
        dof_2 = 10
    else: 
        dof_2 = float(dof2)
    
    
    n = 100
    
    if dropdown == 'exponential' and dropdown2 == 'exponential':
        
        X = np.linspace(0,5,n)
        Y = np.linspace(0,5,n) 
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_exponential_exponential(x,y)
                Z.append(entry)

        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_1),
                            yaxis_title= 'Exponential({})'.format(lambda_2),),
                          )
    

    elif dropdown == 'exponential' and dropdown2 == 'normal':

        X = np.linspace(0,mu_2+2*std_2,n) #exponential
        Y = np.linspace(mu_2 - 3*std_2, mu_2 + 3*std_2, n) #normal
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_exponential_normal(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_1),
                            yaxis_title= 'Normal({}, {})'.format(mu_2, std_2**2),),
                          )
    
    
    elif dropdown == 'normal' and dropdown2 == 'exponential':
        
        X = np.linspace(mu_1 - 3*std_1, mu_1 + 3*std_1, n) #normal
        Y = np.linspace(0,mu_1+2*std_1,n) #exponential
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_normal_exponential(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_2),
                            yaxis_title= 'Normal({}, {})'.format(mu_1, std_1**2),),
                          )

    
    elif dropdown == 'normal' and dropdown2 == 'normal':
        
        Z = []
        
        min_X = mu_1-3*std_1
        max_X = mu_1+3*std_1
        min_Y = mu_2-3*std_2
        max_Y = mu_2+3*std_2
        range_X = max_X - min_X
        range_Y = max_Y - min_Y
        length = max(range_X,range_Y)
        
        X = np.linspace(mu_1-length/2, mu_1+length/2,n)
        Y = np.linspace(mu_2-length/2, mu_2+length/2,n)
        
        for x in X:
            for y in Y:
                entry = norm.pdf(x,loc = mu_1, scale = std_1)*norm.pdf(y, loc= mu_2, scale = std_2)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Normal({}, {})'.format(mu_1, std_1**2),
                            yaxis_title= 'Normal({}, {})'.format(mu_2, std_2**2),),
                          )
    
    
    if dropdown == 'exponential' and dropdown2 == 'gamma':
        
        X = np.linspace(0,10,n) # exponential
        Y = np.linspace(0,10,n) # gamma 
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_exponential_gamma(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_1),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_2, rate_2),),
                          )
    
    
    if dropdown == 'gamma' and dropdown2 == 'gamma':
        #max(1/rate_1, 10)
        #max(1/rate_2, 10)
        
        X = np.linspace(0,10,n)
        Y = np.linspace(0,10,n)
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_gamma_gamma(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Gamma ({}, {})'.format(shape_1, rate_1),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_2, rate_2),),
                          )
    

    if dropdown == 'gamma' and dropdown2 == 'exponential':
        
        X = np.linspace(0,10,n) # gamma
        Y = np.linspace(0,10,n) # exponential
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_exponential_gamma(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_2),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_1, rate_1),),
                          )
    

    if dropdown == 'gamma' and dropdown2 == 'normal':
        
        X = np.linspace(0,mu_2+ 2*std_2,n) # gamma 
        Y = np.linspace(mu_2 - 3*std_2, mu_2 + 3*std_2, n) # normal
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_gamma_normal(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Gamma ({}, {})'.format(shape_1, rate_1),
                            yaxis_title= 'Normal ({}, {})'.format(mu_2, std_2),),
                          )
    
    
    if dropdown == 'normal' and dropdown2 == 'gamma':

        X = np.linspace(mu_1 - 3*std_1, mu_1 + 3*std_1, n) # normal 
        Y = np.linspace(0,mu_1+ 2*std_1,n) # gamma
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_normal_gamma(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Normal ({}, {})'.format(mu_1, std_1),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_2, rate_2),),
                          )
       
    if dropdown == 't-distribution' and dropdown2 == 't-distribution':

        X = np.linspace(-5, 5, n)  
        Y = np.linspace(-5, 5, n)  
        Z = []
        
        for x in X:
            for y in Y:
                entry = t.pdf(x, df = dof_1) * t.pdf(y, df = dof_2)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='t-distribution ({})'.format(dof_1),
                            yaxis_title= 't-distribution  ({})'.format(dof_2),),
                          )
        
    if dropdown == 't-distribution' and dropdown2 == 'normal':

        X = np.linspace(min(-5, mu_2 - 3*std_2), max(5,mu_2 + 3*std_2), n) # t-distribution
        Y = np.linspace(min(-5,mu_2 - 3*std_2), max(mu_2 + 3*std_2,5), n) # normal 
        Z = []
        
        for x in X:
            for y in Y:
                entry = t.pdf(x, df = dof_1)*norm.pdf(y, loc = mu_2, scale = std_2)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='t-distribution ({})'.format(dof_1),
                            yaxis_title= 'Normal ({}, {})'.format(mu_2, std_2),),
                          )

    
    if dropdown == 't-distribution' and dropdown2 == 'exponential':

        X = np.linspace(-5, 5, n) # t-distribution
        Y = np.linspace(0,10,n) # exponential
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_t_dist_exponential(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='t-distribution ({})'.format(dof_1),
                            yaxis_title= 'Exponential ({})'.format(lambda_2),),
                          )
        
    if dropdown == 't-distribution' and dropdown2 == 'gamma':

        X = np.linspace(-5, 5, n) # t-distribution 
        Y = np.linspace(0,10,n) # gamma
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_t_dist_gamma(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='t-distribution ({})'.format(dof_1),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_2, rate_2),),
                          )
        
    if dropdown == 'normal' and dropdown2 == 't-distribution':

        X = np.linspace(mu_1 - 3*std_1, mu_1 + 3*std_1, n) # normal 
        Y = np.linspace(min(-5,mu_1 - 3*std_1),max(5,mu_1+ 2*std_1),n) # t-distribution
        Z = []
        
        for x in X:
            for y in Y:
                entry = norm.pdf(x, loc = mu_1, scale = std_1)*t.pdf(y, df = dof_2)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Normal ({}, {})'.format(mu_1, std_1),
                            yaxis_title= 't-distribution ({})'.format(dof_2),),
                          )
        
    if dropdown == 'exponential' and dropdown2 == 't-distribution':

        X = np.linspace(0, 5, n) # exponential 
        Y = np.linspace(-5,5,n) # t-distribution
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_exponential_t_dist(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Exponential ({})'.format(lambda_1),
                            yaxis_title= 't-distribution ({})'.format(dof_2),),
                          )
        
    if dropdown == 'gamma' and dropdown2 == 't-distribution':

        X = np.linspace(0, 10, n) # gamma 
        Y = np.linspace(-5,5,n) # t-distribution
        Z = []
        
        for x in X:
            for y in Y:
                entry = joint_pdf_gamma_t_dist(x,y)
                Z.append(entry)
        
        fig2.update_layout(
                          scene = dict(
                            xaxis_title='Gamma ({}, {})'.format(shape_1, rate_1),
                            yaxis_title= 't-distribution ({})'.format(dof_2),),
                          )
        
    Z = np.asarray(Z)
    X = np.repeat(list(X),n)
    Y = np.array(list(Y)*n)
    fig2.update_traces(x=X, y=Y, z=Z, intensity=np.array(Z))
    
    return fig2

# Showing the output

# To open the window automatically
port = 8050
def open_browser():
	webbrowser.open_new("http://localhost:{}".format(port))

if __name__ == '__main__':
    Timer(1, open_browser).start();
    app.run_server(debug=True)
    




    
