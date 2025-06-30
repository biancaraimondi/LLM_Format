#!/usr/bin/env python3
"""
A simple Lisp interpreter in Python
Supports basic arithmetic, conditionals, variables, functions, and lists
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
    # Add spaces around parentheses and split
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    return s.split()

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
    tokens = tokenize(s)
    exp = parse(tokens)
    return eval_exp(exp, env)

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
        "(+ 1 2 3)",
        "(* 2 3 4)",
        "(- 10 3)",
        "(/ 15 3)",
        "(define x 10)",
        "(+ x 5)",
        "(define square (lambda (x) (* x x)))",
        "(square 5)",
        "(defun cube (x) (* x x x))",
        "(cube 3)",
        "(defun add-two (x y) (+ x y))",
        "(add-two 10 20)",
        "(if (> 5 3) 'yes 'no)",
        "(defun factorial (n) (if (= n 0) 1 (* n (factorial (- n 1)))))",
        "(factorial 5)",
        "(list 1 2 3 4)",
        "(car (list 1 2 3))",
        "(cdr (list 1 2 3))",
        "(cons 0 (list 1 2 3))",
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