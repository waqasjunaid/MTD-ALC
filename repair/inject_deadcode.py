# repair/inject_deadcode.py
import random

# Expanded set of realistic-looking dead code snippets
DEAD_CODE_SNIPPETS = [
    # Useless assignments / redundant code
    "int dummy = 0; dummy += 1;",
    "char buf[10]; memset(buf, 0, sizeof(buf));",
    "int x = rand() % 100; if (x > 1000) { /* impossible */ }",

    # Fake dangerous calls inside dead branches
    "if (0) { strcpy(dest, src); }",
    "if (false) { gets(input); }",
    "if (1 == 2) { sprintf(buf, fmt, args); }",

    # Harmless but looks suspicious
    "printf(\"debug: %d\\n\", some_var);",
    "FILE *fp = fopen(\"/dev/null\", \"w\"); fclose(fp);",
    "int unused = strlen(\"hello\");",

    # More subtle dead code
    "int safe_counter = 0; safe_counter++;",
    "double temp = 3.14; temp *= 1.0;",
    "/* TODO: remove later */ int placeholder = 42;",

    # Fake buffer operations
    "char temp_buf[64]; strncpy(temp_buf, \"safe\", sizeof(temp_buf)-1);",
    "int len = strlen(safe_str); if (len < 0) { /* impossible */ }",

    # Fake loop or condition
    "for (int i = 0; i < 0; i++) { do_nothing(); }",
    "while (false) { risky_operation(); }"
]


def inject_dead_code(code: str) -> str:
    """
    Inject one random dead code snippet at the beginning of the code.
    """
    dc = random.choice(DEAD_CODE_SNIPPETS)
    # Add some randomness: sometimes add extra newline or comment
    if random.random() < 0.3:
        dc = "// dead code: " + dc
    return dc + "\n\n" + code