#!/usr/bin/env python3
"""
A simple Lisp interpreter in Python
Supports basic arithmetic, conditionals, variables, functions, and lists
Enhanced with setq and let* special forms
"""

import re
import operator as op
from typing import Any, List, Dict, Callable, Union

# Types
Symbol = str
Number = Union[int, float]
Atom = Union[Symbol, Number]
Exp = Union[Atom, List]

class LispError(Exception):
    pass

class Environment:
    """An environment: a dict of {'var': val} pairs, with an outer Environment."""
    
    def __init__(self, params=(), args=(), outer=None):
        self.data = dict(zip(params, args))
        self.outer = outer
    
    def find(self, var):
        """Find the innermost Environment where var appears."""
        if var in self.data:
            return self
        elif self.outer is not None:
            return self.outer.find(var)
        else:
            raise LispError(f"Undefined variable: {var}")
    
    def get(self, var):
        """Get the value of a variable."""
        return self.find(var).data[var]
    
    def set(self, var, val):
        """Set a variable in this environment."""
        self.data[var] = val

class Procedure:
    """A user-defined Lisp procedure."""
    
    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env
    
    def __call__(self, *args):
        if len(args) != len(self.params):
            raise LispError(f"Expected {len(self.params)} arguments, got {len(args)}")
        new_env = Environment(self.params, args, self.env)
        return eval_exp(self.body, new_env)

def standard_env():
    """Create an environment with standard Lisp procedures."""
    env = Environment()
    env.data.update({
        # Arithmetic
        '+': lambda *args: sum(args),
        '-': lambda x, *args: x - sum(args) if args else -x,
        '*': lambda *args: eval('*'.join(map(str, args))) if args else 1,
        '/': lambda x, y: x / y,
        '//': lambda x, y: x // y,
        '%': lambda x, y: x % y,
        
        # Comparison
        '=': lambda x, y: x == y,
        '<': lambda x, y: x < y,
        '>': lambda x, y: x > y,
        '<=': lambda x, y: x <= y,
        '>=': lambda x, y: x >= y,
        
        # Logic
        'and': lambda *args: all(args),
        'or': lambda *args: any(args),
        'not': lambda x: not x,
        
        # List operations
        'cons': lambda x, y: [x] + (y if isinstance(y, list) else [y]),
        'car': lambda x: x[0] if x else None,
        'cdr': lambda x: x[1:] if len(x) > 1 else [],
        'list': lambda *args: list(args),
        'length': len,
        'append': lambda *args: sum(args, []),
        'null?': lambda x: x == [],
        'list?': lambda x: isinstance(x, list),
        
        # Type predicates
        'number?': lambda x: isinstance(x, (int, float)),
        'symbol?': lambda x: isinstance(x, str) and x not in ['#t', '#f'],
        
        # Constants
        '#t': True,
        '#f': False,
        'nil': [],
        
        # I/O
        'print': print,
    })
    
    # Compute multiplication properly
    def multiply(*args):
        result = 1
        for arg in args:
            result *= arg
        return result
    env.data['*'] = multiply
    
    return env

def tokenize(s):
    """Convert a string into a list of tokens."""
    # Remove comments (everything after semicolon until end of line)
    lines = s.split('\n')
    cleaned_lines = []
    for line in lines:
        # Find semicolon that's not inside a string
        in_string = False
        for i, char in enumerate(line):
            if char == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
            elif char == ';' and not in_string:
                line = line[:i]
                break
        cleaned_lines.append(line)
    
    s = '\n'.join(cleaned_lines)
    
    # Add spaces around parentheses and split
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    return [token for token in s.split() if token]

def parse(tokens):
    """Parse a list of tokens into an expression."""
    if not tokens:
        raise LispError("Unexpected EOF")
    
    token = tokens.pop(0)
    if token == '(':
        exp = []
        while tokens and tokens[0] != ')':
            exp.append(parse(tokens))
        if not tokens:
            raise LispError("Missing closing parenthesis")
        tokens.pop(0)  # Remove ')'
        return exp
    elif token == ')':
        raise LispError("Unexpected closing parenthesis")
    else:
        return atom(token)

def atom(token):
    """Convert a token to an atomic value."""
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token

def eval_exp(exp, env):
    """Evaluate an expression in an environment."""
    if isinstance(exp, str):  # Variable reference
        return env.get(exp)
    elif not isinstance(exp, list):  # Constant literal
        return exp
    elif not exp:  # Empty list
        return []
    
    # Special forms
    op = exp[0]
    
    if op == 'quote':
        return exp[1]
    elif op == 'if':
        (_, test, conseq, alt) = exp
        return eval_exp(conseq if eval_exp(test, env) else alt, env)
    elif op == 'define':
        (_, var, val) = exp
        env.set(var, eval_exp(val, env))
        return var
    elif op == 'set!':
        (_, var, val) = exp
        env.find(var).data[var] = eval_exp(val, env)
        return var
    elif op == 'setq':
        # setq is like set! but doesn't require the variable to exist first
        # In many Lisp dialects, setq can create new variables in the current scope
        if len(exp) < 3:
            raise LispError("setq requires at least 2 arguments")
        
        # Handle multiple variable assignments: (setq x 1 y 2 z 3)
        if len(exp) % 2 == 0:
            raise LispError("setq requires an even number of arguments (var val pairs)")
        
        result = None
        for i in range(1, len(exp), 2):
            var = exp[i]
            val = eval_exp(exp[i + 1], env)
            env.set(var, val)  # setq creates in current environment
            result = val
        return result
    elif op == 'let*':
        # let* binds variables sequentially, so later bindings can use earlier ones
        if len(exp) < 3:
            raise LispError("let* requires at least 2 arguments")
        
        bindings = exp[1]
        body = exp[2:]
        
        # Create new environment extending the current one
        new_env = Environment(outer=env)
        
        # Process bindings sequentially
        for binding in bindings:
            if not isinstance(binding, list) or len(binding) != 2:
                raise LispError("let* binding must be a list of [var val]")
            var, val_exp = binding
            val = eval_exp(val_exp, new_env)  # Use new_env so we can reference previous bindings
            new_env.set(var, val)
        
        # Evaluate body in the new environment
        result = None
        for expr in body:
            result = eval_exp(expr, new_env)
        return result
    elif op == 'lambda':
        (_, params, body) = exp
        return Procedure(params, body, env)
    elif op == 'defun':
        (_, name, params, body) = exp
        proc = Procedure(params, body, env)
        env.set(name, proc)
        return name
    elif op == 'begin':
        val = None
        for exp in exp[1:]:
            val = eval_exp(exp, env)
        return val
    elif op == 'cond':
        for clause in exp[1:]:
            if len(clause) == 1:  # else clause
                return eval_exp(clause[0], env)
            test, result = clause
            if eval_exp(test, env):
                return eval_exp(result, env)
        return None
    else:  # Procedure call
        proc = eval_exp(op, env)
        args = [eval_exp(arg, env) for arg in exp[1:]]
        if callable(proc):
            try:
                return proc(*args)
            except TypeError as e:
                raise LispError(f"Error calling {op}: {e}")
        else:
            raise LispError(f"{op} is not a procedure")

def lisp_eval(s, env=None):
    """Evaluate a Lisp expression from a string."""
    if env is None:
        env = standard_env()
    
    # Handle multi-line expressions
    lines = s.strip().split('\n')
    results = []
    current_expr = ""
    paren_count = 0
    
    for line in lines:
        # Skip empty lines and comment-only lines
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
            
        current_expr += line + "\n"
        
        # Count parentheses to determine if expression is complete
        for char in line:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
        
        # If parentheses are balanced, evaluate the expression
        if paren_count == 0 and current_expr.strip():
            try:
                tokens = tokenize(current_expr)
                if tokens:  # Only parse if there are tokens
                    exp = parse(tokens)
                    result = eval_exp(exp, env)
                    results.append(result)
            except Exception:
                # If multi-line parsing fails, try single expression
                tokens = tokenize(s)
                exp = parse(tokens)
                return eval_exp(exp, env)
            current_expr = ""
    
    # If we have a single result, return it directly
    # If multiple results, return the last one (like most Lisp interpreters)
    return results[-1] if results else None

def repl():
    """Run a Read-Eval-Print Loop."""
    global_env = standard_env()
    print("Simple Lisp Interpreter")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            inp = input("lisp> ")
            if inp.strip().lower() == 'quit':
                break
            if inp.strip():
                result = lisp_eval(inp, global_env)
                if result is not None:
                    print(result)
        except (LispError, Exception) as e:
            print(f"Error: {e}")

# Example usage and tests
if __name__ == "__main__":
    # Run some example expressions
    examples = [
        "(+ 1 2 3)  ; Simple addition",
        "(* 2 3 4)  ; Multiplication",
        "(- 10 3)   ; Subtraction", 
        "(/ 15 3)   ; Division",
        "(define x 10)  ; Define a variable",
        "(+ x 5)    ; Use the variable",
        "; Testing setq - can set multiple variables at once",
        "(setq a 5 b 10 c 15)",
        "(+ a b c)  ; Should be 30",
        "; setq can also set single variables",
        "(setq result (* a b))",
        "result      ; Should be 50",
        "; Testing let* - sequential binding",
        "(let* ((x 5) (y (* x 2)) (z (+ x y))) z)",  # Should be 15
        "; Another let* example",
        "(let* ((a 3) (b (+ a 2)) (c (* a b))) (list a b c))",  # Should be (3 5 15)",
        "; Define a function using lambda",
        "(define square (lambda (x) (* x x)))",
        "(square 5)",
        "; Define a function using defun - more convenient!",
        "(defun cube (x) (* x x x))",
        "(cube 3)",
        "(defun add-two (x y) (+ x y))  ; Function with two parameters",
        "(add-two 10 20)",
        "(if (> 5 3) 'yes 'no)  ; Conditional expression",
        "; Using let* in a function",
        "(defun quadratic (x)",
        "  (let* ((x2 (* x x)) (x3 (* x x2))) (+ x x2 x3)))",
        "(quadratic 3)  ; 3 + 9 + 27 = 39",
        "; Recursive function definition",
        "(defun factorial (n)",
        "  (if (= n 0)      ; Base case",
        "      1            ; Return 1 if n is 0", 
        "      (* n (factorial (- n 1)))))  ; Recursive case",
        "(factorial 5)  ; Calculate 5!",
        "(list 1 2 3 4)     ; Create a list",
        "(car (list 1 2 3)) ; Get first element",
        "(cdr (list 1 2 3)) ; Get rest of list",
        "(cons 0 (list 1 2 3))  ; Add element to front",
    ]
    
    print("Running example expressions:")
    print("=" * 40)
    
    env = standard_env()
    for example in examples:
        try:
            result = lisp_eval(example, env)
            print(f"{example} => {result}")
        except Exception as e:
            print(f"{example} => Error: {e}")
    
    print("\n" + "=" * 40)
    print("Starting REPL (type 'quit' to exit):")
    repl()