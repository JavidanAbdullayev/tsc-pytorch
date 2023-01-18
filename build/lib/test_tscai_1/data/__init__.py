
import os

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def subtract(a, b):
    return a - b

def divide(a, b):
    return a / b

def exp(a, b):
    return a**b

def display_path():
    directory = os.getcwd()
    print(directory)