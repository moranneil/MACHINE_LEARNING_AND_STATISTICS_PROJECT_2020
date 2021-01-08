import sklearn.linear_model as lin
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns




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
<head>
    <title>Power Predictor Results</title>

    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>

    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

</head>
<table class="table">
    <tr><td><h3><br>Linear Regression Predicted Power Output from Wind Speed of """ + speed_input + " kph = " + str(algorithms.receive_speed_from_webpage(float(speed_input))) + "kW</h3></td></tr><tr><td><h3>Neural Network Using RAW Dataset Predicted Power Output from Wind Speed of " + speed_input + " kph = " + str(algorithms.receive_speed_from_webpage_Neural(float(speed_input))) + """kW</h3></td></tr><tr><td><h3>""" "Neural Network Using Cleaned Dataset Predicted Power Output from Wind Speed of " + speed_input + " kph = " + str(algorithms.receive_speed_from_webpage_Neural_Clean(float(speed_input))) + """kW</h3></td></tr>
</table>
<br><button onclick="backToInputPage()">Back to Input Page</button>

<script>
    function backToInputPage() {
        window.history.back();
    }
</script>
"""

if __name__ == "__main__":
    app.run(debug=True)

