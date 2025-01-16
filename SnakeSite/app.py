from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('base.html')  # We'll use base.html as our main and only template

if __name__ == '__main__':
    app.run(debug=True)