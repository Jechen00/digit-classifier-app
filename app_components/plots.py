#####################################
# Packages & Dependencies
#####################################
import param
import panel as pn

import torch
import numpy as np
import plotly.graph_objects as go

from . import canvas
from app_utils import styles
from model_training import data_setup, model


#####################################
# Plotly Panels
#####################################
PLOTLY_CONFIGS = {
    'displayModeBar': True, 'displaylogo': False,
    'modeBarButtonsToRemove': ['autoScale', 'lasso', 'select', 
                               'toImage', 'pan', 'zoom', 'zoomIn', 'zoomOut']
}

class PlotPanels(param.Parameterized):
    canvas_info = param.ClassSelector(class_ = canvas.Canvas)
    
    def __init__(self, mod_path, mod_kwargs, **params):
        super().__init__(**params)
        self.class_labels = np.arange(0, 10)
        self.cnn_mod = model.TinyVGG(**mod_kwargs)
        self.cnn_mod.load_state_dict(torch.load(mod_path, map_location = 'cpu'))
        
        self.img_pane = pn.pane.Plotly(
            name = 'image_plot',
            config = PLOTLY_CONFIGS,
            sizing_mode = 'stretch_both',
            margin = 0,
        )

        self.prob_pane = pn.pane.Plotly(
            name = 'prob_plot',
            config = PLOTLY_CONFIGS,
            sizing_mode = 'stretch_both',
            margin = 0
        )
        
        self.pred_txt = pn.pane.HTML(
            styles = {'margin':'0rem', 'color':styles.CLRS['pred_txt'], 
                      'font-size':styles.FONTSIZES['pred_txt'],
                      'font-family':styles.FONTFAMILY}
        )

        # Initialize plotly figures
        self._update_prediction()

        # Set up watchers thta update based on data URI changes
        self.canvas_info.param.watch(self._update_prediction, 'uri')

    def _update_prediction(self, *event):
        self._update_preprocessed_tensor()
        self._update_pred_txt()
        self._update_img_plot()
        self._update_prob_plot()

    def _update_preprocessed_tensor(self):
        # Check if uri is non-empty
        if self.canvas_info.uri:
            self.input_img = data_setup.mnist_preprocess(self.canvas_info.uri)

            self.cnn_mod.eval() # Set CNN to eval & inferene mode
            with torch.inference_mode():
                pred_logits = self.cnn_mod(self.input_img.unsqueeze(0))
                self.pred_probs = torch.softmax(pred_logits, dim = 1)[0].numpy()
                self.pred_label = np.argmax(self.pred_probs)
        else:
            self.input_img = torch.zeros((28, 28))
            self.pred_probs = np.zeros(10)
            self.pred_label = None

    def _update_pred_txt(self):
        if self.canvas_info.uri:
            pred, prob = self.pred_label, f'{self.pred_probs[self.pred_label]:.3f}'
        else:
            pred, prob = 'N/A', 'N/A'

        self.pred_txt.object = f'''
            <div style="text-align: left;">
                <b>Prediction:</b> {pred}
                </br>
                <b>Probability:</b> {prob}
            </div>
        '''

    def _update_prob_plot(self):
        # Marker fill and outline color for bar plot
        mkr_clrs = [styles.CLRS['base_bar']] * len(self.class_labels)
        mkr_line_clrs = [styles.CLRS['base_bar_line']] * len(self.class_labels)
        if self.pred_label is not None:
            mkr_clrs[self.pred_label] = styles.CLRS['pred_bar']
            mkr_line_clrs[self.pred_label] = styles.CLRS['pred_bar_line']
            
        fig = go.Figure()
        # Bar plot
        fig.add_trace(
            go.Bar(x = self.class_labels, y = self.pred_probs, 
                   marker_color = mkr_clrs, marker_line_color = mkr_line_clrs,
                   marker_line_width = 1.5, showlegend = False,
                   text = self.pred_probs, textposition = 'outside',
                   textfont = dict(color = styles.CLRS['plot_txt'],
                                   size = styles.FONTSIZES['plot_bar_txt'], family = styles.FONTFAMILY), 
                   texttemplate = '%{text:.3f}', 
                   customdata = self.pred_probs * 100,
                   hoverlabel_font = dict(family = styles.FONTFAMILY),
                   hovertemplate = '<b>Class Label:</b> %{x}' +
                                   '<br><b>Probability:</b> %{customdata:.2f} %' +
                                   '<extra></extra>'
            )
        )
        # Used to fix axis limits
        fig.add_trace(
            go.Scatter(
                x = [0.5, 0.5], y = [0.1, 1],
                marker = dict(color = 'rgba(0, 0, 0, 0)', size = 10),
                mode = 'markers', 
                hoverinfo = 'skip', 
                showlegend = False
            )
        )
        fig.update_yaxes(
            title = dict(text = 'Prediction Probability', standoff = 0,
                         font = dict(color = styles.CLRS['plot_txt'],
                                     size = styles.FONTSIZES['plot_labels'], 
                                     family = styles.FONTFAMILY)),
            tickfont = dict(size = styles.FONTSIZES['plot_ticks'], 
                            family = styles.FONTFAMILY),
            dtick = 0.1, ticks = 'outside', ticklen = 0,
            gridcolor = styles.CLRS['prob_plot_grid']
        )
        fig.update_xaxes(
            title = dict(text = 'Class Label', standoff = 6,
                         font = dict(color = styles.CLRS['plot_txt'],
                                     size = styles.FONTSIZES['plot_labels'], 
                                     family = styles.FONTFAMILY)),
            dtick = 1, tickfont = dict(size = styles.FONTSIZES['plot_ticks'], 
                                       family = styles.FONTFAMILY),
        )
        fig.update_layout(
            paper_bgcolor = styles.CLRS['prob_plot_bg'],
            plot_bgcolor = styles.CLRS['prob_plot_bg'],
            margin = dict(l = 60, r = 0, t = 5, b = 45),
        )

        self.prob_pane.object = fig
        
    def _update_img_plot(self):
        img_np = self.input_img.squeeze().numpy()

        if self.pred_label is not None:
            zmin, zmax = np.min(img_np), np.max(img_np)
        else:
            zmin, zmax = 0, 1

        fig = go.Figure(
            data = go.Heatmap(
                z = img_np,
                colorscale = 'gray',
                showscale = False,
                zmin = zmin,
                zmax = zmax,
                hoverlabel_font = dict(family = styles.FONTFAMILY),
                hovertemplate = '<b>Pixel Position:</b> (%{x}, %{y})' +
                                '<br><b>Pixel Value:</b> %{z:.3f}' + 
                                '<extra></extra>'
            )
        )

        fig.update_yaxes(autorange = 'reversed') 
        fig.update_layout(
            plot_bgcolor = styles.CLRS['img_plot_bg'],
            margin = dict(l = 0, r = 0, t = 0, b = 0),
            xaxis = dict(showticklabels = False),
            yaxis = dict(showticklabels = False),
        )

        self.img_pane.object = fig
