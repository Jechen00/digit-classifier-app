#####################################
# Packages & Dependencies
#####################################
import panel as pn
import os, yaml
from panel.viewable import Viewer

from app_components import canvas, plots
from app_utils import styles

pn.extension('plotly')
FILE_PATH = os.path.dirname(__file__)


################################################
# Digit Classifier Layout
################################################
class DigitClassifier(Viewer):
    '''
    Builds and displays the UI for the classifier application. 

    Args:
        mod_path (str): The absolute path to the saved TinyVGG model
        mod_kwargs (dict): A dictionary containing the keyword-arguments for the TinyVGG model.
                           This should have the keys: num_blks, num_convs, in_channels, hidden_channels, and num_classes
    '''

    def __init__(self, mod_path: str, mod_kwargs: dict, **params):
        self.canvas = canvas.Canvas(sizing_mode = 'stretch_both', 
                                    styles = {'border':'black solid 0.15rem'})
        
        self.clear_btn = pn.widgets.Button(name = 'Clear', 
                                           sizing_mode = 'stretch_width',
                                           stylesheets = [styles.BTN_STYLESHEET])

        self.plot_panels = plots.PlotPanels(canvas_info = self.canvas, mod_path = mod_path, mod_kwargs = mod_kwargs)
        
        super().__init__(**params)
        self.github_logo = pn.pane.PNG(
            object = FILE_PATH + '/assets/github-mark-white.png',
            alt_text = 'GitHub Repo',
            link_url = 'https://github.com/Jechen00/digit-classifier-app',
            height = 70,
            styles = {'margin':'0'}
        )
        self.controls_col = pn.FlexBox(
            self.github_logo,
            self.clear_btn, 
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
            sizing_mode = 'stretch_width',
            styles = {'height':'60%'}
        )

        self.prob_row =  pn.FlexBox(self.plot_panels.prob_pane,
                                    sizing_mode = 'stretch_width',
                                    styles = {'height':'40%', 
                                              'border':'solid 0.15rem black'})

        self.page_info = pn.pane.HTML(
            object = f'''
                <style>
                    .link {{
                        color: rgb(29, 161, 242);
                        text-decoration: none;
                        transition: text-decoration 0.2s ease;
                    }}

                    .link:hover {{
                        text-decoration: underline;
                    }}
                </style>

                <div style="text-align:center; font-size:{styles.FONTSIZES['sidebar_title']};margin-top:0.2rem">
                    <b>Digit Classifier</b>
                </div>

                <div style="padding:0 2.5% 0 2.5%; text-align:left; font-size:{styles.FONTSIZES['sidebar_txt']}; width: 100%;">
                    <hr style="height:2px; background-color:rgb(200, 200, 200); border:none; margin-top:0">

                    <p style="margin:0">
                        This is a handwritten digit classifier that uses a <i>convolutional neural network (CNN)</i>
                        to make predictions. The architecture of the model is a scaled-down version of 
                        the <i>Visual Geometry Group (VGG)</i> architecture from the paper:
                        <a href="https://arxiv.org/pdf/1409.1556" 
                           class="link" 
                           target="_blank" 
                           rel="noopener noreferrer">
                        Very Deep Convolutional Networks for Large-Scale Image Recognition</a>.
                    </p>
                    </br>
                    <p style="margin:0">
                        <b>How To Use:</b> Draw a digit (0-9) on the canvas 
                        and the model will produce a prediction for it in real time.
                        Prediction probabilities (or confidences) for each digit are displayed in the bar chart, 
                        reflecting the model's softmax output distribution.
                        To the right of the canvas, you'll also find the transformed input image, i.e. the canvas drawing after undergoing  
                        <a href="https://paperswithcode.com/dataset/mnist"
                           class="link" 
                           target="_blank" 
                           rel="noopener noreferrer">
                        MNIST preprocessing</a>.  
                        This input image represents what the model receives prior to feature extraction and classification.
                    </p>
                </div>
                <div style="margin-left: 5px; margin-top: 72px">
                    <a href="https://github.com/Jechen00"
                    class="link"
                    target="blank"
                    rel="noopener noreferrer"
                    style="font-size: {styles.FONTSIZES['made_by_txt']}; color: {styles.CLRS['made_by_txt']};">
                        Made by Jeff Chen
                    </a>
                </div>
            ''',
            styles = {'margin':' 0rem', 'color': styles.CLRS['sidebar_txt'], 
                      'width': '19.7%', 'height': '100%',
                      'font-family': styles.FONTFAMILY,
                      'background-color': styles.CLRS['sidebar'],
                      'overflow-y':'scroll',
                      'border': 'solid 0.15rem black'}
        )

        self.classifier_content = pn.FlexBox(
            self.img_row,
            self.prob_row,
            gap = '0.5%',
            flex_direction = 'column',
            flex_wrap = 'nowrap',
            sizing_mode = 'stretch_height',
            styles = {'width': '80%'}
        )

        self.page_content = pn.FlexBox(
            self.page_info,
            self.classifier_content,
            gap = '0.3%',
            flex_direction = 'row',
            justify_content = 'space-around',
            align_items = 'center',
            flex_wrap = 'nowrap',
            styles = {
                'height':'100%',
                'width':'100vw',
                'padding': '1%',
                'min-width': '1200px',
                'min-height': '600px',
                'max-width': '3600px',
                'max-height': '1800px',
                'background-color': styles.CLRS['page_bg']
            },
        )

        # This is mainly used to ensure there is always have a grey background
        self.page_layout = pn.FlexBox(
            self.page_content,
            justify_content = 'center',
            flex_wrap = 'nowrap',
            sizing_mode = 'stretch_both',
            styles = {
                'min-width': 'max-content',
                'background-color': styles.CLRS['page_bg'],
            }
        )
        # Set up on-click event with clear button and the canvas
        self.clear_btn.on_click(self.canvas.toggle_clear)
        
    def __panel__(self):
        '''
        Returns the main layout of the application to be rendered by Panel.
        '''
        return self.page_layout


def create_app():
    '''
    Creates the application, ensuring that each user gets a different instance of digit_classifier.
    Mostly used to keep things away from a global scope.
    '''
    # Used to serve with panel serve in command line
    save_dir = FILE_PATH + '/saved_models'
    base_name = 'tiny_vgg'

    mod_path = f'{save_dir}/{base_name}_model.pth' # Path to the saved model state dict
    settings_path = f'{save_dir}/{base_name}_settings.yaml' # Path to the saved model kwargs

    # Load in model kwargs
    with open( settings_path, 'r') as f:
        loaded_settings = yaml.load(f, Loader = yaml.FullLoader)

    mod_kwargs = loaded_settings['mod_kwargs']

    digit_classifier = DigitClassifier(mod_path = mod_path, mod_kwargs = mod_kwargs)
    return digit_classifier

################################################
# Serve App
################################################
# Used to serve with panel serve in command line
create_app().servable(title = 'CNN Digit Classifier')