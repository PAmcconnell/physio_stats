import base64
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import webbrowser
from threading import Timer
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename):
    try:
        # Decode the base64 file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Embed the HTML content in an iframe
        srcdoc = decoded.decode('utf-8')
        return html.Div([
            html.H5(filename),
            html.Iframe(srcDoc=srcdoc, style={"width": "100%", "height": "800px"})  # Adjusted height
        ])
    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")
        return html.Div([
            f"There was an error processing file {filename}"
        ])

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('upload-data', 'style')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        try:
            children = [
                parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)]
            return children, {'display': 'none'}  # Hide the upload component
        except Exception as e:
            logging.error(f"Error updating output: {e}")
            return html.Div([
                "There was an error processing the file"
            ]), {'display': 'block'}
    return [], {'display': 'block'}  # Show the upload component

# Function to open the web browser
def open_browser():
    try:
        webbrowser.open_new("http://127.0.0.1:8050/")
    except Exception as e:
        logging.error(f"Error opening browser: {e}")

# Run the app
if __name__ == '__main__':
    Timer(1, open_browser).start()
    try:
        app.run_server(debug=True)
    except Exception as e:
        logging.error(f"Error running the app: {e}")
