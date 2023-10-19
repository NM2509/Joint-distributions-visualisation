#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 18:22:17 2022

@author: anastasiamalakhova 
"""

#######################
# Importing libraries #
#######################

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

#################
# Initial graph #
#################

number_of_points = 100
X_axis = np.linspace(0,5,number_of_points)
Y_axis = np.linspace(0,5,number_of_points)
lambda_x_axis = 1
lambda_y_axis = 1

def joint_pdf_default(x,y):
    f_xy = lambda_x_axis*math.exp(-lambda_x_axis*x)*lambda_y_axis*math.exp(-lambda_y_axis*y)
    return f_xy

Z_axis = []

for x in X_axis:
    for y in Y_axis:
        entry = joint_pdf_default(x,y)
        Z_axis.append(entry)
Z_axis = np.asarray(Z_axis)

X_axis = np.repeat(list(X_axis),number_of_points)
Y_axis = np.array(list(Y_axis)*number_of_points)


# Initialising figures in Dash

app = dash.Dash(__name__,  external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

colors = {'background': 'rgb(223,245,249)'}


figure_to_display = go.Figure(go.Mesh3d(x=X_axis,y=Y_axis,
                          z=np.array(Z_axis),opacity=0.7, 
                          colorscale="greens", intensity=np.array(Z_axis), 
                          visible = True))

figure_to_display.update_layout(paper_bgcolor = colors['background'], 
                  plot_bgcolor = colors['background'], 
                  width=800,
                  height=700,
                  autosize=False,
                  margin=dict(t=70, b=70, l=0, r=0),
                  scene = dict(
                    xaxis_title='Exponential({})'.format(lambda_x_axis),
                    yaxis_title= 'Exponential({})'.format(lambda_y_axis),
                    zaxis_title='Joint pdf'),
                  )

##############
# App layout #
##############

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
        figure=figure_to_display,)],
        style={'margin-left': '3vw', 'margin-top': '1vw', 'margin-right': '5vw', 'display': 'grid',  'grid-column-start': '2','grid-row-start': '2', 'grid-row-end': '40'}),
      
    ],
             
    # style of the layout    
    style={'backgroundColor':colors['background'], 'display': 'grid', 'grid-template-columns': '30% 70%'})
             

             
# Template - background of the graph
pio.templates["custom_dark"] = go.layout.Template(
    layout=go.Layout(
        colorway=['#ff0000', '#00ff00', '#0000ff']))
      

pio.templates['custom_dark']['layout']['yaxis']['gridcolor'] = 'rgb(223,245,249)'
pio.templates['custom_dark']['layout']['xaxis']['gridcolor'] = 'rgb(223,245,249)'
figure_to_display.layout.template = 'custom_dark'

#########################
# Callback - parameters #
#########################

# Define a mapping of component IDs to the distributions 

component_visibility_1 = {
    'name1': ['normal'],
    'mu1_': ['normal'],
    'name2': ['normal'],
    'std1_': ['normal'],
    'name3': ['exponential'],
    'lambda1_': ['exponential'],
    'name7': ['gamma'],
    'shape1': ['gamma'],
    'name8': ['gamma'],
    'rate1': ['gamma'],
    'name11': ['t-distribution'],
    'dof1': ['t-distribution']
}

component_visibility_2 = {
    'name4': ['normal'],
    'mu2_': ['normal'],
    'name5': ['normal'],
    'std2_': ['normal'],
    'name6': ['exponential'],
    'lambda2_': ['exponential'],
    'name9': ['gamma'],
    'shape2': ['gamma'],
    'name10': ['gamma'],
    'rate2': ['gamma'],
    'name12': ['t-distribution'],
    'dof2': ['t-distribution']
}

def determine_visibility(visibility_state, component_id, component_visibility):
    """Determine the visibility based on the selected distribution and component ID."""
    if visibility_state in component_visibility.get(component_id, []):
        return {'display': 'block'}
    return {'display': 'none'}

# Callback for components related to dropdown_first_graph
@app.callback(
    [Output(component_id, 'style') for component_id in component_visibility_1.keys()],
    [Input(component_id='dropdown_first_graph', component_property='value')]
)
def callback_1(visibility_state_1):
    return [determine_visibility(visibility_state_1, component_id, component_visibility_1) for component_id in component_visibility_1.keys()]

# Callback for components related to dropdown_second_graph
@app.callback(
    [Output(component_id, 'style') for component_id in component_visibility_2.keys()],
    [Input(component_id='dropdown_second_graph', component_property='value')]
)
def callback_2(visibility_state_2):
    return [determine_visibility(visibility_state_2, component_id, component_visibility_2) for component_id in component_visibility_2.keys()]


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
        f_xy = lambda_x_axis*math.exp(-lambda_x_axis*x)*lambda_y_axis*math.exp(-lambda_y_axis*y)
        return f_xy
    
    def joint_pdf_exponential_normal(x,y):
        f_xy = lambda_x_axis*math.exp(-lambda_x_axis*x)*(1/(std_2*math.sqrt(2*math.pi)))*math.exp(-0.5*(((y-mu_2)/std_2)**2))
        return f_xy
    
    def joint_pdf_normal_exponential(x,y):
        f_xy = (1/(std_1*math.sqrt(2*math.pi)))*math.exp(-0.5*(((x-mu_1)/std_1)**2))*lambda_y_axis*math.exp(-lambda_y_axis*y)
        return f_xy
    
    def joint_pdf_exponential_gamma(x,y):
        f_xy = lambda_x_axis*math.exp(-lambda_x_axis*x)*((rate_2**shape_2)*(y**(shape_2-1))*math.exp(-rate_2*y))/math.factorial(shape_2-1)
        return f_xy
    
    def joint_pdf_gamma_exponential(x,y):
        f_xy = (((rate_1**shape_1)*(x**(shape_1-1))*math.exp(-rate_1*x))/math.factorial(shape_1-1))*(lambda_y_axis*math.exp(-lambda_y_axis*y))
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
        f_xy = (gamma((dof_1+1)/2)*(1+((x**2)/dof_1))**(-0.5*(dof_1+1)))/(math.sqrt(math.pi*dof_1)*gamma(dof_1/2))*(lambda_y_axis*math.exp(-lambda_y_axis*y))
        return f_xy
                    
    def joint_pdf_gamma_t_dist(x,y):
        f_xy = (((rate_1**shape_1)*(x**(shape_1-1))*math.exp(-rate_1*x))/math.factorial(shape_1-1))*(gamma((dof_2+1)/2)*(1+((y**2)/dof_2))**(-0.5*(dof_2+1)))/(math.sqrt(math.pi*dof_2)*gamma(dof_2/2))
        return f_xy
        
    def joint_pdf_exponential_t_dist(x,y):
        f_xy = lambda_x_axis*math.exp(-lambda_x_axis*x)*(gamma((dof_2+1)/2)*(1+((y**2)/dof_2))**(-0.5*(dof_2+1)))/(math.sqrt(math.pi*dof_2)*gamma(dof_2/2))
        return f_xy
            
    
    def convert_or_default(value, default_value):
        return default_value if value == '' else float(value)

    # Parameters and their default values
    params_defaults = [
        (mu1_value, 10),
        (std1_value, 3),
        (mu2_value, 10),
        (std2_value, 3),
        (lambda1_value, 1),
        (lambda2_value, 1),
        (shape1, 2),
        (rate1, 0.5),
        (shape2, 2),
        (rate2, 0.5),
        (dof1, 10),
        (dof2, 10)
    ]
    
    # Apply conversion function
    mu_1, std_1, mu_2, std_2, lambda_x_axis, lambda_y_axis, shape_1, rate_1, shape_2, rate_2, dof_1, dof_2 = [convert_or_default(value, default) for value, default in params_defaults]

    
    
    n = 100
    
    if dropdown == 'exponential' and dropdown2 == 'exponential':
        
        X_axis = np.linspace(0,5,n)
        Y_axis = np.linspace(0,5,n) 
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_exponential_exponential(x,y)
                Z_axis.append(entry)

        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_x_axis),
                            yaxis_title= 'Exponential({})'.format(lambda_y_axis),),
                          )
    

    elif dropdown == 'exponential' and dropdown2 == 'normal':

        X_axis = np.linspace(0,mu_2+2*std_2,n) #exponential
        Y_axis = np.linspace(mu_2 - 3*std_2, mu_2 + 3*std_2, n) #normal
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_exponential_normal(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_x_axis),
                            yaxis_title= 'Normal({}, {})'.format(mu_2, std_2**2),),
                          )
    
    
    elif dropdown == 'normal' and dropdown2 == 'exponential':
        
        X_axis = np.linspace(mu_1 - 3*std_1, mu_1 + 3*std_1, n) #normal
        Y_axis = np.linspace(0,mu_1+2*std_1,n) #exponential
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_normal_exponential(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_y_axis),
                            yaxis_title= 'Normal({}, {})'.format(mu_1, std_1**2),),
                          )

    
    elif dropdown == 'normal' and dropdown2 == 'normal':
        
        Z_axis = []
        
        min_X = mu_1-3*std_1
        max_X = mu_1+3*std_1
        min_Y = mu_2-3*std_2
        max_Y = mu_2+3*std_2
        range_X = max_X - min_X
        range_Y = max_Y - min_Y
        length = max(range_X,range_Y)
        
        X_axis = np.linspace(mu_1-length/2, mu_1+length/2,n)
        Y_axis = np.linspace(mu_2-length/2, mu_2+length/2,n)
        
        for x in X_axis:
            for y in Y_axis:
                entry = norm.pdf(x,loc = mu_1, scale = std_1)*norm.pdf(y, loc= mu_2, scale = std_2)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Normal({}, {})'.format(mu_1, std_1**2),
                            yaxis_title= 'Normal({}, {})'.format(mu_2, std_2**2),),
                          )
    
    
    if dropdown == 'exponential' and dropdown2 == 'gamma':
        
        X_axis = np.linspace(0,10,n) # exponential
        Y_axis = np.linspace(0,10,n) # gamma 
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_exponential_gamma(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_x_axis),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_2, rate_2),),
                          )
    
    
    if dropdown == 'gamma' and dropdown2 == 'gamma':
        #max(1/rate_1, 10)
        #max(1/rate_2, 10)
        
        X_axis = np.linspace(0,10,n)
        Y_axis = np.linspace(0,10,n)
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_gamma_gamma(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Gamma ({}, {})'.format(shape_1, rate_1),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_2, rate_2),),
                          )
    

    if dropdown == 'gamma' and dropdown2 == 'exponential':
        
        X_axis = np.linspace(0,10,n) # gamma
        Y_axis = np.linspace(0,10,n) # exponential
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_exponential_gamma(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Exponential({})'.format(lambda_y_axis),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_1, rate_1),),
                          )
    

    if dropdown == 'gamma' and dropdown2 == 'normal':
        
        X_axis = np.linspace(0,mu_2+ 2*std_2,n) # gamma 
        Y_axis = np.linspace(mu_2 - 3*std_2, mu_2 + 3*std_2, n) # normal
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_gamma_normal(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Gamma ({}, {})'.format(shape_1, rate_1),
                            yaxis_title= 'Normal ({}, {})'.format(mu_2, std_2),),
                          )
    
    
    if dropdown == 'normal' and dropdown2 == 'gamma':

        X_axis = np.linspace(mu_1 - 3*std_1, mu_1 + 3*std_1, n) # normal 
        Y_axis = np.linspace(0,mu_1+ 2*std_1,n) # gamma
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_normal_gamma(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Normal ({}, {})'.format(mu_1, std_1),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_2, rate_2),),
                          )
       
    if dropdown == 't-distribution' and dropdown2 == 't-distribution':

        X_axis = np.linspace(-5, 5, n)  
        Y_axis = np.linspace(-5, 5, n)  
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = t.pdf(x, df = dof_1) * t.pdf(y, df = dof_2)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='t-distribution ({})'.format(dof_1),
                            yaxis_title= 't-distribution  ({})'.format(dof_2),),
                          )
        
    if dropdown == 't-distribution' and dropdown2 == 'normal':

        X_axis = np.linspace(min(-5, mu_2 - 3*std_2), max(5,mu_2 + 3*std_2), n) # t-distribution
        Y_axis = np.linspace(min(-5,mu_2 - 3*std_2), max(mu_2 + 3*std_2,5), n) # normal 
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = t.pdf(x, df = dof_1)*norm.pdf(y, loc = mu_2, scale = std_2)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='t-distribution ({})'.format(dof_1),
                            yaxis_title= 'Normal ({}, {})'.format(mu_2, std_2),),
                          )

    
    if dropdown == 't-distribution' and dropdown2 == 'exponential':

        X_axis = np.linspace(-5, 5, n) # t-distribution
        Y_axis = np.linspace(0,10,n) # exponential
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_t_dist_exponential(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='t-distribution ({})'.format(dof_1),
                            yaxis_title= 'Exponential ({})'.format(lambda_y_axis),),
                          )
        
    if dropdown == 't-distribution' and dropdown2 == 'gamma':

        X_axis = np.linspace(-5, 5, n) # t-distribution 
        Y_axis = np.linspace(0,10,n) # gamma
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_t_dist_gamma(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='t-distribution ({})'.format(dof_1),
                            yaxis_title= 'Gamma ({}, {})'.format(shape_2, rate_2),),
                          )
        
    if dropdown == 'normal' and dropdown2 == 't-distribution':

        X_axis = np.linspace(mu_1 - 3*std_1, mu_1 + 3*std_1, n) # normal 
        Y_axis = np.linspace(min(-5,mu_1 - 3*std_1),max(5,mu_1+ 2*std_1),n) # t-distribution
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = norm.pdf(x, loc = mu_1, scale = std_1)*t.pdf(y, df = dof_2)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Normal ({}, {})'.format(mu_1, std_1),
                            yaxis_title= 't-distribution ({})'.format(dof_2),),
                          )
        
    if dropdown == 'exponential' and dropdown2 == 't-distribution':

        X_axis = np.linspace(0, 5, n) # exponential 
        Y_axis = np.linspace(-5,5,n) # t-distribution
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_exponential_t_dist(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Exponential ({})'.format(lambda_x_axis),
                            yaxis_title= 't-distribution ({})'.format(dof_2),),
                          )
        
    if dropdown == 'gamma' and dropdown2 == 't-distribution':

        X_axis = np.linspace(0, 10, n) # gamma 
        Y_axis = np.linspace(-5,5,n) # t-distribution
        Z_axis = []
        
        for x in X_axis:
            for y in Y_axis:
                entry = joint_pdf_gamma_t_dist(x,y)
                Z_axis.append(entry)
        
        figure_to_display.update_layout(
                          scene = dict(
                            xaxis_title='Gamma ({}, {})'.format(shape_1, rate_1),
                            yaxis_title= 't-distribution ({})'.format(dof_2),),
                          )
        
        
    Z_axis = np.asarray(Z_axis)
    X_axis = np.repeat(list(X_axis),n)
    Y_axis = np.array(list(Y_axis)*n)
    figure_to_display.update_traces(x=X_axis, y=Y_axis, z=Z_axis, intensity=np.array(Z_axis))
    
    return figure_to_display

######################
# Showing the output #
######################

# To open the window automatically
port = 8050
def open_browser():
	webbrowser.open_new("http://localhost:{}".format(port))

if __name__ == '__main__':
    Timer(1, open_browser).start();
    app.run_server(debug=True)
