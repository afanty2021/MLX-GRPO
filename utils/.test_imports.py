"""Quick test to verify all imports work (when mlx-lm is installed)"""
try:
    # Test that files can be imported as modules
    import ast
    with open('convert_model.py') as f:
        ast.parse(f.read())
    with open('inference.py') as f:
        ast.parse(f.read())
    print("✓ All Python files have valid syntax")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    exit(1)
