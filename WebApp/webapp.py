import sklearn.linear_model as lin
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


#https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/models.ipynb
#https://stackoverflow.com/questions/30336324/seaborn-load-dataset
#https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/



import algorithms


from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def webpage():
    return render_template('homepage.html')

@app.route('/', methods=['POST'])
def webpage_post():
    speed_input = request.form['speed_input']
    print("Wind Speed Entered: ",speed_input)
    return """ 
    <table><tr><th><h2><br>Linear Regression Predicted Power Output from Wind Speed of """ + speed_input + " kph = " + str(algorithms.receive_speed_from_webpage(float(speed_input))) + "kW</h2></th></tr><tr><td><h2>""" "Neural Network Predicted Power Output from Wind Speed of " + speed_input + " kph = " + str(algorithms.receive_speed_from_webpage_Neural(speed_input)) + """kW</h2></td></tr></tr></table>
    <br><button onclick="backToInputPage()">Back to Input Page</button>

<script>
function backToInputPage() {
  window.history.back();
}
</script>
    """

if __name__ == "__main__":
    app.run(debug=True)

