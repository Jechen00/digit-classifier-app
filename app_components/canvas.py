#####################################
# Packages
#####################################
import param
import panel as pn
from panel.reactive import ReactiveHTML
from panel.viewable import Viewer

from model_training import data_setup


#####################################
# Canvas
#####################################
class Canvas(ReactiveHTML):
    '''
    Reference: https://panel.holoviz.org/how_to/custom_components/examples/canvas_draw.html
    '''
    uri = param.String()
    clear = param.Boolean(default = False)
    
    _template = '''
        <canvas
          id="canvas"
          style="width: 100%; height: 100%"
          height=400px
          width=400px
          onmousedown="${script('start')}"
          onmousemove="${script('draw')}"
          onmouseup="${script('end')}"
          onmouseleave="${script('end')}">
        </canvas>
    '''
    
    _scripts = {
        'render': '''
            state.ctx = canvas.getContext('2d');
            state.ctx.fillStyle = '#FFFFFF';
            state.ctx.fillRect(0, 0, canvas.width, canvas.height);
            state.ctx.lineWidth = 30;
            state.ctx.strokeStyle = '#000000';
            state.ctx.lineJoin = 'round';
            state.ctx.lineCap = 'round';

            // Helper to normalize mouse coordinates
            state.getCoords = function(e) {
                const rect = canvas.getBoundingClientRect();
                return {
                    x: (e.clientX - rect.left) * (canvas.width / rect.width),
                    y: (e.clientY - rect.top) * (canvas.height / rect.height)
                };
            };
        ''',
        
       'start': '''
            if (state.isDrawing) return;
            state.isDrawing = true;
            const pos = state.getCoords(event);
            state.ctx.beginPath();
            state.ctx.moveTo(pos.x, pos.y);
        ''',
        
        'draw': '''
            if (!state.isDrawing) return;
            const pos = state.getCoords(event);
            state.ctx.lineTo(pos.x, pos.y);
            state.ctx.stroke();
            data.uri = canvas.toDataURL('image/png');
        ''',
        
        'end': '''
            if (!state.isDrawing) return;      // Early return if already not drawing
            state.isDrawing = false;
        ''',
        
        'clear': '''
            state.ctx.fillStyle = '#FFFFFF';
            state.ctx.fillRect(0, 0, canvas.width, canvas.height);
            data.uri = '';
        '''
    }

    def toggle_clear(self, *event):
        self.clear = not self.clear