## 1) `ZOF_CLI.py`

import math
import sys
from typing import Callable, List, Dict, Tuple, Any

# --- Utility: safe eval for user function expression ---

MATH_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
MATH_NAMES.update({
    'abs': abs,
    'pow': pow
})


def make_function(expr: str) -> Callable[[float], float]:
    """Return a function f(x) that evaluates the expression safely.
    Expression must be a valid Python expression using `x` and math functions.
    Example: "x**3 - 2*x - 5" or "sin(x) - x/2"
    """
    expr = expr.strip()

    def f(x: float) -> float:
        local = dict(MATH_NAMES)
        local['x'] = x
        try:
            return float(eval(expr, {"__builtins__": None}, local))
        except Exception as e:
            raise ValueError(f"Error evaluating expression at x={x}: {e}")

    return f


# --- Root-finding methods ---

def bisection(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int):
    logs = []
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Bisection requires f(a) and f(b) to have opposite signs.")
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        err = abs(b - a) / 2.0
        logs.append((i, a, b, c, fa, fb, fc, err))
        if abs(fc) < tol or err < tol:
            return c, err, i, logs
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2.0, abs(b - a) / 2.0, max_iter, logs


def regula_falsi(f, a: float, b: float, tol: float, max_iter: int):
    logs = []
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Regula Falsi requires f(a) and f(b) to have opposite signs.")
    c = a
    for i in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        err = abs(fc)
        logs.append((i, a, b, c, fa, fb, fc, err))
        if abs(fc) < tol:
            return c, err, i, logs
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, abs(fc), max_iter, logs


def secant(f, x0: float, x1: float, tol: float, max_iter: int):
    logs = []
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            raise ZeroDivisionError('Division by zero in secant step')
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        err = abs(x2 - x1)
        logs.append((i, x0, x1, x2, f0, f1, err))
        if err < tol:
            return x2, err, i, logs
        x0, x1 = x1, x2
    return x1, abs(x1 - x0), max_iter, logs


def newton_raphson(f, df, x0: float, tol: float, max_iter: int):
    logs = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError('Zero derivative encountered in Newton-Raphson')
        x_new = x - fx / dfx
        err = abs(x_new - x)
        logs.append((i, x, x_new, fx, dfx, err))
        if err < tol:
            return x_new, err, i, logs
        x = x_new
    return x, err, max_iter, logs


def fixed_point(g, x0: float, tol: float, max_iter: int):
    logs = []
    x = x0
    for i in range(1, max_iter + 1):
        x_new = g(x)
        err = abs(x_new - x)
        logs.append((i, x, x_new, err))
        if err < tol:
            return x_new, err, i, logs
        x = x_new
    return x, err, max_iter, logs


def modified_secant(f, x0: float, delta: float, tol: float, max_iter: int):
    logs = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        denom = f(x + delta * x) - fx
        if denom == 0:
            raise ZeroDivisionError('Denominator zero in modified secant')
        x_new = x - (delta * x * fx) / denom
        err = abs(x_new - x)
        logs.append((i, x, x_new, fx, err))
        if err < tol:
            return x_new, err, i, logs
        x = x_new
    return x, err, max_iter, logs


# --- CLI user interaction ---

def print_menu():
    print("\nZOF Solver - Choose a method:")
    print("1. Bisection")
    print("2. Regula Falsi (False Position)")
    print("3. Secant")
    print("4. Newton-Raphson")
    print("5. Fixed Point Iteration")
    print("6. Modified Secant")
    print("0. Exit")


def read_float(prompt: str, default=None):
    while True:
        s = input(prompt)
        if s == '' and default is not None:
            return default
        try:
            return float(s)
        except ValueError:
            print('Enter a valid number')


def main():
    print("Zero-of-Functions (ZOF) CLI Solver")
    while True:
        print_menu()
        choice = input("Select method (0-6): ").strip()
        if choice == '0':
            print('Goodbye')
            break
        if choice not in [str(i) for i in range(1, 7)]:
            print('Invalid choice')
            continue

        expr = input("Enter f(x) as a Python expression (use 'x', math funcs allowed):\n e.g. x**3 - 2*x - 5\n f(x) = ")
        f = make_function(expr)

        tol = read_float('Tolerance (e.g. 1e-6) [default 1e-6]: ', 1e-6)
        max_iter = int(read_float('Max iterations [default 50]: ', 50))

        try:
            if choice == '1':
                a = read_float('Left endpoint a: ')
                b = read_float('Right endpoint b: ')
                root, ferr, iters, logs = bisection(f, a, b, tol, max_iter)
                print('\nIteration logs: (i, a, b, c, f(a), f(b), f(c), err)')
                for row in logs:
                    print(row)

            elif choice == '2':
                a = read_float('Left endpoint a: ')
                b = read_float('Right endpoint b: ')
                root, ferr, iters, logs = regula_falsi(f, a, b, tol, max_iter)
                print('\nIteration logs: (i, a, b, c, f(a), f(b), f(c), err)')
                for row in logs:
                    print(row)

            elif choice == '3':
                x0 = read_float('x0: ')
                x1 = read_float('x1: ')
                root, ferr, iters, logs = secant(f, x0, x1, tol, max_iter)
                print('\nIteration logs: (i, x0, x1, x2, f(x0), f(x1), err)')
                for row in logs:
                    print(row)

            elif choice == '4':
                df_expr = input("Enter f'(x) as Python expression (derivative).\nIf you don't want to type it, enter 'auto' to approximate numerically: ")
                x0 = read_float('Initial guess x0: ')
                if df_expr.strip().lower() == 'auto':
                    def df_approx(x):
                        h = 1e-6
                        return (f(x + h) - f(x - h)) / (2 * h)
                    df = df_approx
                else:
                    df = make_function(df_expr)
                root, ferr, iters, logs = newton_raphson(f, df, x0, tol, max_iter)
                print('\nIteration logs: (i, x, x_new, f(x), f\'(x), err)')
                for row in logs:
                    print(row)

            elif choice == '5':
                g_expr = input("Enter g(x) for Fixed Point (x = g(x)): ")
                g = make_function(g_expr)
                x0 = read_float('Initial guess x0: ')
                root, ferr, iters, logs = fixed_point(g, x0, tol, max_iter)
                print('\nIteration logs: (i, x, x_new, err)')
                for row in logs:
                    print(row)

            elif choice == '6':
                x0 = read_float('Initial guess x0: ')
                delta = read_float('Delta (fraction, e.g. 1e-3) [default 1e-3]: ', 1e-3)
                root, ferr, iters, logs = modified_secant(f, x0, delta, tol, max_iter)
                print('\nIteration logs: (i, x, x_new, f(x), err)')
                for row in logs:
                    print(row)

            print('\nFinal result:')
            print(f'Estimated root = {root}')
            print(f'Final error estimate = {ferr}')
            print(f'Iterations = {iters}\n')

        except Exception as e:
            print('Error during method:', e)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')

