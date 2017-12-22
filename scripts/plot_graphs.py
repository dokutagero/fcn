import numpy as np
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Legend, Span
from bokeh.io import export_png
import pandas as pd

df = pd.read_csv('fcn16s_40e.csv')
train_miu = df['train_total/mean_iu'].dropna().values
valid_miu = df['valid/mean_iu'].dropna().values
epochs = list(range(train_miu.shape[0]))
train_loss = np.log(df['train_total/loss'].dropna().values)
test_loss = np.log(df['valid/loss'].dropna().values)

output_file('file.png')
p = figure(plot_width=600, plot_height=600)
train_plot = p.line(epochs, train_miu, color='red')
test_plot = p.line(epochs, valid_miu, color='blue')
legend=Legend(items=[('train', [train_plot]), ('test', [test_plot])], location=(0, 500)) 
p.add_layout(legend, 'right')
p.toolbar.logo = None
p.toolbar_location = None
p.yaxis.axis_label = "Mean IoU"
p.xaxis.axis_label = 'Epochs'
p.title.text = 'FCN16s Mean IoU'
export_png(p, filename='file.png')
# Train/Test loss
output_file('loss.png')
p = figure(plot_width=600, plot_height=600)
train_plot = p.line(epochs, train_loss, color='red')
test_plot = p.line(epochs, test_loss, color='blue')
legend=Legend(items=[('train', [train_plot]), ('test', [test_plot])], location=(0, 500)) 
p.add_layout(legend, 'right')
# selected_epoch = Span(location=28, dimension='height', line_color='green', line_dash='dashed')
# p.add_layout(selected_epoch)
p.toolbar.logo = None
p.toolbar_location = None
p.yaxis.axis_label = "Log loss"
p.xaxis.axis_label = "Epochs"
p.title.text = 'FCN16s Log loss'
export_png(p, filename='loss.png')
