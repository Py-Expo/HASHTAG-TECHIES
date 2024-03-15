from flask import Flask,request,render_template,url_for,redirect
from flask_mysqldb import MySQL
from detect import detector

app=Flask(__name__)

app.config['MYSQL_HOST']= "localhost"
app.config['MYSQL_DB']= "flask"
app.config['MYSQL_USER']= "root"
app.config['MYSQL_PASSWORD']= "root"
app.config['MYSQL_CURSORCLASS']="DictCursor"
app.secret_key="myapp"
conn = MySQL(app)

@app.route('/')
def login():
    return render_template("home .html")

'''@app.route('/login', methods = ['POST', 'GET'])
def login():
    if request.method  == 'POST' or 'GET':
        user_name1 = request.form['user_name']
        password1 = request.form['password']
        con=conn.connection.cursor()
        sql = "select * from login WHERE username= %s and  password=%s"
        result=con.execute(sql,(user_name1,password1))
        if result:
            con.connection.commit()
            con.close()
            return redirect(url_for('appl'))
        else:
            return "Invalid Username or Password"
               
                
    return render_template("home.html") '''


if __name__=="__main__":
    app.run(debug=False)