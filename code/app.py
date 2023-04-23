from flask import Flask, render_template, request, jsonify
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import plotly.figure_factory as ff
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import probplot
from pyhive import hive


app = Flask(__name__)


# Connect to Hive server
conn = hive.connect(host='localhost', port=10000, database='default')
# Execute Hive query and fetch results
cursor = conn.cursor()
cursor.execute('SELECT * FROM sales')
result_set = cursor.fetchall()
# Create Pandas DataFrame from result set
df = pd.DataFrame(result_set[1:], columns=[col[0] for col in cursor.description])
# Close Hive connection
conn.close()

# Load the data from csv
#df = pd.read_csv("sales.csv")

def generate_url(fig):
    buffer = io.BytesIO()
    fig.figure.savefig(buffer, format='png')
    buffer.seek(0)    
    # Encode the plot to base64
    plot_url = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return plot_url

# Define endpoints for each visualization type
@app.route('/scatter', methods=['POST'])
def scatter():
    x_col = request.form['x-col']
    y_col = request.form['y-col']
    fig = px.scatter(df, x=x_col, y=y_col)
    plot_url = fig.to_html(full_html=False)
    return jsonify({'plot_url': plot_url})


@app.route('/line', methods=['POST'])
def line():
    x_col = request.form['x-col']
    y_col = request.form['y-col']
    fig = px.line(df, x=x_col, y=y_col)
    fig.update_layout(
        title="Line Plot",
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    plot_html = fig.to_html(full_html=False)
    return jsonify({'plot_url': plot_html})


@app.route('/bar', methods=['POST'])
def bar():
    x_col = request.form['x-col']
    y_col = request.form['y-col']
    fig = px.bar(df, x=x_col, y=y_col)
    plot_url = fig.to_html(full_html=False)
    return jsonify({'plot_url': plot_url})

@app.route('/distribution-matrix', methods=['POST'])
def distribution_matrix():    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        if i >= len(axes): # Check if index i is greater than or equal to the length of axes
            break # Exit the loop to avoid accessing an out-of-bounds index
        sns.histplot(df[col], ax=axes[i])
        axes[i].set_xlabel(col)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.suptitle('Distribution Plots')
    plot_url = generate_url(fig)
    return jsonify({'plot_url': plot_url})
    

@app.route('/count', methods=['POST'])
def count_plot():
    col = request.form['col']
    counts = df[col].value_counts()
    data = [go.Bar(x=counts.index, y=counts.values)]
    layout = go.Layout(title=f"Count Plot for {col}")
    fig = go.Figure(data=data, layout=layout)
    plot_url = pio.to_html(fig, full_html=False)      
    return jsonify({'plot_url': plot_url})

@app.route('/heatmap', methods=['POST'])
def heatmap():
    colorscale = request.form['colorScale']
    corr = np.round(df.corr(),2)    
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale=colorscale))
    fig.update_layout(title='Correlation Heatmap')
    plot_url = pio.to_html(fig, full_html=False)
    return jsonify({'plot_url': plot_url})


@app.route('/distribution', methods=['POST'])
def distribution(): 
    # generate some sample data
    col = request.form['col']
    data = df[col]
    # create histogram figure using plotly express
    fig = px.histogram(data, nbins=30, opacity=0.8)
    # add a fitted normal distribution curve to the figure
    mu, std = norm.fit(data)
    x = np.linspace(np.min(data), np.max(data), 1000)
    y = norm.pdf(x, mu, std)
    fig.add_trace(px.line(x=x, y=y).data[0])
    # update the figure layout with title and axis labels
    fig.update_layout(title='Distribution of Data', xaxis_title='Values', yaxis_title='Frequency')
    # generate the plotly URL
    plot_url = fig.to_html(full_html=False)
    # return the URL as a JSON response
    return jsonify({'plot_url': plot_url})
    

@app.route('/probability', methods=['POST'])
def probability_plot():
    col = request.form['col']
    # get column data
    column_data = df[col]    
    # create probability plot
    probplot_data = probplot(column_data, plot=None)
    x = probplot_data[0][0]
    y = probplot_data[0][1]    
    # create plotly figure
    fig = px.scatter(x=x, y=y, trendline='ols', trendline_color_override='red')
    fig.update_layout(title=f"Probability plot for '{col}' column")    
    # get plotly HTML
    plot_url = fig.to_html(full_html=False)    
    return jsonify({'plot_url': plot_url})


@app.route('/pie', methods=['POST'])
def pie_chart():
    col = request.form['col']
    # group data by column name and count the number of occurrences
    data = df.groupby(col).size().reset_index(name='count')    
    # create plotly pie chart
    fig = px.pie(data, values='count', names=col)
    fig.update_layout(title=f"Pie chart for '{col}' column")
    
    # get plotly HTML
    plot_url = fig.to_html(full_html=False)
    return jsonify({'plot_url': plot_url})

@app.route('/box', methods=['POST'])
def box_plot():
    x_col = request.form['x-col']
    y_col = request.form['y-col']
    # create plotly box plot
    fig = px.box(df, x=x_col, y=y_col)
    fig.update_layout(title=f"Box plot for '{y_col}' by '{x_col}'")    
    # get plotly HTML
    plot_url = fig.to_html(full_html=False)    
    return jsonify({'plot_url': plot_url})


# Define the homepage
@app.route('/')
def index():
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    return render_template('index.html', columns=columns,numeric_cols=numeric_cols)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
