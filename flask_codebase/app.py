from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', fig=None)  # This will render your HTML page

if __name__ == '__main__':
    app.run(debug=True)