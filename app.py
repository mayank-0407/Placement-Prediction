import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

# Creating app
app=Flask(__name__)

# Loading the pkl file
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    gender=int(request.form.get("gender"))
    sscp=int(request.form.get("sscp"))
    sscb=int(request.form.get("sscb"))
    hscp=int(request.form.get("hscp"))
    hscb=int(request.form.get("hscb"))
    hscs=int(request.form.get("hscs"))
    cgpa=int(request.form.get("cgpa"))
    degreet=int(request.form.get("degreet"))
    workex=int(request.form.get("workex"))
    etestp=int(request.form.get("etestp"))
    specialisation=int(request.form.get("specialisation"))
    masters=int(request.form.get("masters"))
    # features=[int(x) for x in string_features]
    # print(features)
    # features=[np.array(features)]
    # new_features=features[np.newaxis,:]
    prediction=model.predict([[gender,sscp,sscb,hscp,hscb,hscs,cgpa,degreet,workex,etestp,specialisation,masters]])
    print(prediction)
    if(prediction[0]):
        return render_template("index.html",prediction_text="You will be Placed")
    return render_template("index.html",prediction_text="You will be Not Placed")

if __name__=="__main__":
    app.run(debug=True)

    # 1,67.0,1,91.0,1,1,58.0,2,0,55.0,1,58,8