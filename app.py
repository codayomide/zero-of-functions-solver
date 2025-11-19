## 2) `app.py` (Flask web GUI)
from flask import Flask, render_template, request, redirect, url_for
import math
from typing import Callable

app = Flask(__name__)

# reuse make_function and methods from CLI (copy-paste for a single-file app)
MATH_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
MATH_NAMES.update({'abs': abs, 'pow': pow})

def make_function(expr: str):
    expr = expr.strip()
    def f(x: float):
        local = dict(MATH_NAMES)
        local['x'] = x
        return float(eval(expr, {"__builtins__": None}, local))
    return f

# (For brevity, implement a simple wrapper for Bisection and Secant and Newton; other methods can be
# added similarly. The CLI has full implementations to copy if you prefer full parity.)

def bisection(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    logs = []
    if fa * fb > 0:
        raise ValueError('f(a) and f(b) must have opposite signs')
    for i in range(1, max_iter+1):
        c = (a+b)/2
        fc = f(c)
        err = abs(b-a)/2
        logs.append({'i':i, 'a':a,'b':b,'c':c,'fa':fa,'fb':fb,'fc':fc,'err':err})
        if abs(fc) < tol or err < tol:
            return c, err, logs
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a+b)/2, abs(b-a)/2, logs

# Add secant and newton similar to CLI

def secant(f, x0, x1, tol, max_iter):
    logs=[]
    for i in range(1, max_iter+1):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            raise ZeroDivisionError('Division by zero in secant')
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        err = abs(x2-x1)
        logs.append({'i':i,'x0':x0,'x1':x1,'x2':x2,'f0':f0,'f1':f1,'err':err})
        if err < tol:
            return x2, err, logs
        x0, x1 = x1, x2
    return x1, abs(x1-x0), logs

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        method = request.form.get('method')
        expr = request.form.get('expr')
        tol = float(request.form.get('tol') or 1e-6)
        max_iter = int(request.form.get('max_iter') or 50)
        try:
            f = make_function(expr)
            if method == 'bisection':
                a = float(request.form['a'])
                b = float(request.form['b'])
                root, ferr, logs = bisection(f, a, b, tol, max_iter)
                result = {'root': root, 'error': ferr, 'logs': logs}
            elif method == 'secant':
                x0 = float(request.form['x0'])
                x1 = float(request.form['x1'])
                root, ferr, logs = secant(f, x0, x1, tol, max_iter)
                result = {'root': root, 'error': ferr, 'logs': logs}
            elif method == 'newton':
                x0 = float(request.form['x0'])
                df_expr = request.form.get('df_expr')
                if df_expr and df_expr.strip().lower() != 'auto':
                    df = make_function(df_expr)
                    def df_func(x): return df(x)
                else:
                    def df_func(x):
                        h = 1e-6
                        return (f(x+h)-f(x-h))/(2*h)
                # use CLI's newton_raphson logic inline
                x = x0
                logs = []
                for i in range(1, max_iter+1):
                    fx = f(x)
                    dfx = df_func(x)
                    if dfx == 0:
                        raise ZeroDivisionError('Zero derivative')
                    x_new = x - fx/dfx
                    err = abs(x_new-x)
                    logs.append({'i':i,'x':x,'x_new':x_new,'fx':fx,'dfx':dfx,'err':err})
                    if err < tol:
                        root = x_new
                        break
                    x = x_new
                else:
                    root = x
                result = {'root': root, 'error': abs(result['root'] - x) if result else None, 'logs': logs}
        except Exception as e:
            error = str(e)
    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)

