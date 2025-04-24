#####################################
# Fonts & Colors
#####################################
FONTFAMILY = 'Helvetica'

FONTSIZES = {
    'pred_txt': '1.2rem',
    'mod_input_txt': '0.8rem',
    'plot_ticks': 14,
    'plot_labels': 16,
    'plot_bar_txt': 14,
    'btn': '1rem'
}

CLRS = {
    'txt': 'white',
    'base_bar': 'rgb(158, 202, 225)',
    'base_bar_line': 'rgb(8, 48, 107)',
    'pred_bar': 'rgb(240, 140, 140)',
    'pred_bar_line': 'rgb(180, 0, 0)',
    'plot_txt': 'black',
    'prob_plot_bg': 'white',
    'prob_plot_grid': 'rgb(225, 225, 225)',
    'img_plot_bg': 'black',
    'btn_base': 'white',
    'btn_hover': 'rgb(200, 200, 200)',
    'page_bg': 'rgb(150, 150, 150)'
}


#####################################
# Stylesheets
#####################################
BTN_STYLESHEET = f'''
    :host(.solid) .bk-btn {{
        background-color: {CLRS['btn_base']};
        border: black solid 0.1rem;
        border-radius: 0.8rem;
        font-size: {FONTSIZES['btn']};
        padding-top: 0.3rem;
        padding-bottom: 0.3rem;
    }} 
    
    :host(.solid) .bk-btn:hover {{
        background-color: {CLRS['btn_hover']};
    }}
'''