from flask import Flask, render_template, url_for, request, redirect
from genre import *
import warnings
warnings.filterwarnings("ignore")
from werkzeug.utils import secure_filename

with open('mlb_class.pickle', 'rb') as handle:
    pp = pickle.load(handle)


app = Flask(__name__)

@app.route('/')

def hello():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def upload_file():

    if request.method == 'POST':

        # Get the file from post request
        f = request.files['userfile']
    
        # Save the file to ./uploads
	
        img = request.files['userfile']

        img.save("static/"+img.filename)

        text=request.form['message']

        if len(text)==0:
            return render_template('index.html',prediction="Please enter Proper plot")
      
        labels,prob=find_genre("static/"+img.filename,text)
        res = dict(zip(pp, prob)) 

        result_dic={
			'image' : "static/"+img.filename,
            'text' : text,
			'labels' : labels,
           
   }
        return render_template('index.html', results = result_dic)

if __name__=='__main__':
    app.run(threaded=False)