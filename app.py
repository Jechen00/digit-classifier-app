#####################################
# Packages & Dependencies
#####################################
import panel as pn
import os
from panel.viewable import Viewer

from app_components import canvas, plots
from app_utils import styles

pn.extension('plotly')
FILE_PATH = os.path.dirname(__file__)


################################################
# Dashboard - Layout
################################################
class DigitClassifier(Viewer):
    def __init__(self, mod_path, mod_kwargs, **params):
        self.canvas = canvas.Canvas(sizing_mode = 'stretch_both', 
                                    styles = {'border':'black solid 0.15rem'})
        
        self.clear_btn = pn.widgets.Button(name = 'Clear', 
                                           sizing_mode = 'stretch_width',
                                           stylesheets = [styles.BTN_STYLESHEET])

        self.plot_panels = plots.PlotPanels(canvas_info = self.canvas, mod_path = mod_path, mod_kwargs = mod_kwargs)
        
        super().__init__(**params)
        self.github_logo = pn.pane.PNG(
            object = FILE_PATH + '/logos/github-mark-white.png',
            alt_text = 'GitHub Repo',
            link_url = 'https://github.com/Jechen00/digit-classifier-app',
            height = 65,
            styles = {'margin':'0'}
        )
        self.controls_col = pn.FlexBox(
            self.github_logo,
            # pn.Spacer(height = 50),
            self.clear_btn, 
            # pn.Spacer(height = 50),
            self.plot_panels.pred_txt,
            gap = '60px',
            flex_direction = 'column',
            justify_content = 'center',
            align_items = 'center',
            flex_wrap = 'nowrap',
            styles = {'width':'40%', 'height':'100%'}
        )

        self.mod_input_txt = pn.pane.HTML(
            object = '''
                <div>
                    <b>MODEL INPUT</b>
                </div>
            ''',
            styles = {'margin':'0rem', 'padding-left':'0.15rem', 'color':'white', 
                      'font-size':styles.FONTSIZES['mod_input_txt'], 
                      'font-family':styles.FONTFAMILY, 
                      'position':'absolute', 'z-index':'100'}
        )

        self.img_row = pn.FlexBox(
            self.canvas,
            self.controls_col,
            pn.FlexBox(self.mod_input_txt,
                       self.plot_panels.img_pane, 
                       sizing_mode = 'stretch_both',
                       styles = {'border':'solid 0.15rem white'}),
            gap = '1%',
            flex_wrap = 'nowrap',
            flex_direction = 'row',
            justify_content = 'center',
            styles = {'height':'60%', 'width':'80%'}
        )

        self.prob_row =  pn.FlexBox(self.plot_panels.prob_pane,
                                    styles = {'width':'80%', 
                                              'height':'37%', 
                                              'border':'solid 0.15rem black'})

        self.page_content = pn.FlexBox(
            self.img_row,
            self.prob_row,
            gap = '0.5%',
            flex_direction = 'column',
            justify_content = 'center',
            align_items = 'center',
            flex_wrap = 'nowrap',
            styles = {'width': '100%',
                      'height': '100%',
                      'min-width': '1200px',
                      'min-height': '600px',
                      'max-width': '3600px',
                      'max-height': '1800px',
                      'background-color': styles.CLRS['page_bg']},
        )

        self.page_layout = pn.FlexBox(
            self.page_content,
            justify_content = 'center',
            flex_wrap = 'nowrap',
            styles = {
                'height': '100vh',
                'width': '100vw',
                'min-width': 'max-content',
                'background-color': styles.CLRS['page_bg'],
            }
        )
        # Set up on-click event with clear button and the canvas
        self.clear_btn.on_click(self.canvas.toggle_clear)
        
    def __panel__(self):
        return self.page_layout


################################################
# Serve App
################################################
# Used to serve with panel serve in command line
save_dir = './models'
mod_name = 'tiny_vgg_model.pth'
mod_path = f'{save_dir}/{mod_name}'

mod_kwargs = {
    'num_blks': 2,
    'num_convs': 2,
    'in_channels': 1,
    'hidden_channels': 10,
    'num_classes': 10
}

DigitClassifier(mod_path = mod_path, mod_kwargs = mod_kwargs).servable(title = 'CNN Digit Classifier')