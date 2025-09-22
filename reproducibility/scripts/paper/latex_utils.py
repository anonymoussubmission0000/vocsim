# -*- coding: utf-8 -*-
"""
Utility functions for generating and formatting LaTeX tables.
"""
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex

from .table_configs import PRETTY_NAMES, VOCSIM_APPENDIX_S_COLUMN_ORDER
from .data_loader import ConfigManager

logger = logging.getLogger(__name__)


def sanitize(text: Any) -> str:
    """Sanitizes text for LaTeX output, avoiding changes to likely LaTeX commands."""
    if pd.isna(text):
        return "-"
    s_text = str(text)
    # Don't sanitize if it looks like a LaTeX command
    if any(cmd in s_text for cmd in ["\\textbf{", "\\makecell{", "\\multicolumn{", "\\textit{", "\\emph{", "\\texttt{"]):
        return s_text
    
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
        "\\": r"\\", "*": r"$^*$", "<": r"\textless{}", ">": r"\textgreater{}"
    }
    for old, new in replacements.items():
        s_text = s_text.replace(old, new)
    return s_text

def format_number(value: Any, precision: int = 1, is_percentage: bool = False) -> str:
    """Formats a numerical value to a string with specified precision."""
    if pd.isna(value) or value is None:
        return "-"
    if isinstance(value, (int, float, np.number)):
        val_to_format = value * 100 if is_percentage else value
        return f"{val_to_format:.{precision}f}"
    return str(value)

def parse_value_for_comparison(value: Any) -> float:
    """Parses a formatted string (e.g., with bolding) back to a float."""
    if pd.isna(value) or not isinstance(value, str) or value == "-":
        return np.nan
    cleaned = value.replace("\\textbf{", "").replace("}", "").replace("%", "").strip()
    match = re.match(r"^\s*(-?(?:\d+\.\d+|\d+))", cleaned)
    return float(match.group(1)) if match else np.nan

def bold_string(value: Any) -> str:
    """Wraps a string in LaTeX bold command if not already bolded."""
    str_value = str(value)
    if pd.isna(value) or str_value == "-":
        return str_value
    if str_value.startswith("\\textbf{") and str_value.endswith("}"):
        return str_value
    return f"\\textbf{{{str_value}}}"

def bold_best_in_columns(df: DataFrame, columns: List[str], higher_is_better: Dict[str, bool]) -> DataFrame:
    """Bolds the best values in specified columns of a DataFrame."""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        is_higher = higher_is_better.get(col, True)
        numeric_values = result[col].apply(parse_value_for_comparison)
        valid_numeric_values = numeric_values.dropna()
        if valid_numeric_values.empty:
            continue
        best_numeric = valid_numeric_values.max() if is_higher else valid_numeric_values.min()
        for idx in result.index:
            if pd.notna(numeric_values.loc[idx]) and np.isclose(numeric_values.loc[idx], best_numeric):
                result.loc[idx, col] = bold_string(result.loc[idx, col])
    return result

def bold_overall_best_in_group_df(df: pd.DataFrame, columns_to_consider: List[Any], higher_is_better: bool, n_best: int = 1) -> pd.DataFrame:
    """Bolds the top N values across a group of columns in a DataFrame."""
    df_out = df.copy()
    if not columns_to_consider:
        return df_out
    
    all_values = []
    for r_idx in df_out.index:
        for c_idx in columns_to_consider:
            if c_idx not in df_out.columns: continue
            val_str = df_out.loc[r_idx, c_idx]
            numeric_val = parse_value_for_comparison(val_str)
            if pd.notna(numeric_val):
                all_values.append({'val': numeric_val, 'r_idx': r_idx, 'c_idx': c_idx, 'orig_str': val_str})
    
    if not all_values: return df_out
    
    all_values.sort(key=lambda x: x['val'], reverse=higher_is_better)
    
    if n_best > 0 and all_values:
        cutoff_score = all_values[min(n_best, len(all_values)) - 1]['val']
        for item in all_values:
            is_close = np.isclose(item['val'], cutoff_score)
            should_bold = (higher_is_better and (item['val'] > cutoff_score or is_close)) or \
                          (not higher_is_better and (item['val'] < cutoff_score or is_close))
            if should_bold:
                df_out.loc[item['r_idx'], item['c_idx']] = bold_string(item['orig_str'])
    return df_out


def output_latex_table(df: DataFrame, caption: str, label: str, output_file: Path, column_format: Optional[str] = None, notes: Optional[List[str]] = None, is_longtable: bool = False):
    """Generates and writes a LaTeX table (standard or longtable) from a DataFrame."""
    if df.empty:
        logger.warning(f"DataFrame for '{caption}' is empty. Skipping LaTeX output.")
        return

    df_copy = df.copy()
    
    # Sanitize index and columns
    if df_copy.index.name or isinstance(df_copy.index, MultiIndex):
        df_copy.index.names = [sanitize(name) for name in df_copy.index.names]
        if isinstance(df_copy.index, MultiIndex):
            df_copy.index = MultiIndex.from_tuples([tuple(sanitize(level) for level in idx) for idx in df_copy.index])
        else:
            df_copy.index = [sanitize(idx) for idx in df_copy.index]

    if isinstance(df_copy.columns, MultiIndex):
        df_copy.columns.names = [sanitize(name) for name in df_copy.columns.names]
        df_copy.columns = MultiIndex.from_tuples([tuple(sanitize(level) for level in col) for col in df_copy.columns])
    else:
        df_copy.columns = [sanitize(col) for col in df_copy.columns]

    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(lambda x: sanitize(x) if not (isinstance(x, str) and x.startswith("\\")) else x)

    latex_str = df_copy.to_latex(escape=False, na_rep="-", column_format=column_format, index=True, header=True, multirow=True, multicolumn_format="c")
    
    table_env = "longtable" if is_longtable else "table"
    
    latex = [f"\\begin{{{table_env}}}{{'[ht!]' if not is_longtable else ''}}"]
    if not is_longtable:
        latex.append("\\centering")
    latex.append(f"\\caption{{{sanitize(caption)}}}\\label{{{sanitize(label)}}}")
    if is_longtable:
        latex.append("\\\\") # Required for longtable caption
    latex.append("\\small")
    
    if is_longtable:
        # Add longtable headers and footers
        lines = latex_str.splitlines()
        header = "\n".join(lines[2:lines.index("\\midrule") + 1])
        latex.append(header)
        latex.append("\\endfirsthead")
        latex.append(f"\\caption[]{{(Continued) {sanitize(caption)}}}\\\\")
        latex.append(header)
        latex.append("\\endhead")
        latex.append("\\bottomrule")
        latex.append(f"\\multicolumn{{{len(df.columns) + df.index.nlevels}}}{{r}}{{\\textit{{Continued on next page}}}}\\\\")
        latex.append("\\endfoot")
        latex.append("\\bottomrule")
        if notes:
            notes_str = '\\\\\n'.join([sanitize(note) for note in notes])
            latex.append(f"\\multicolumn{{{len(df.columns) + df.index.nlevels}}}{{p{{\\linewidth-2\\tabcolsep}}}}{{\\footnotesize {notes_str}}}\\\\")
        latex.append("\\endlastfoot")
        latex.append("\n".join(lines[lines.index("\\midrule") + 1:-2])) # table body
    else:
        latex.append(latex_str)
        if notes:
            notes_str = '\\\\\n'.join([sanitize(note) for note in notes])
            latex.append("\\smallskip\n\\begin{minipage}{\\textwidth}\\footnotesize")
            latex.append(notes_str)
            latex.append("\\end{minipage}")
        
    latex.append(f"\\end{{{table_env}}}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(latex))
    logger.info(f"LaTeX table '{caption}' saved to {output_file}")


def generate_vocsim_appendix_longtable_latex(df_data: pd.DataFrame, caption_text: str, table_label: str, output_file: Path):
    """
    Generates and writes a VocSim appendix longtable using the STRICT format,
    now compatible with siunitx S columns.
    """
    if df_data.empty:
        logger.warning(f"No data for VocSim Appendix table '{caption_text}'. Skipping file write.")
        return

    expected_cols = 21
    if len(df_data.columns) != expected_cols:
        logger.error(
            f"FATAL: VocSim appendix table expects {expected_cols} data columns, "
            f"but got {len(df_data.columns)}. Halting generation for this table."
        )
        return

    latex_lines = []
    latex_lines.append("% Add to your LaTeX preamble: \\usepackage{longtable, booktabs, siunitx, makecell}")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\setlength{\\tabcolsep}{2pt}")

    s_caption = sanitize(caption_text)
    s_label = sanitize(table_label)

    # Note: table-parse-only used to allow non-numeric content in braces {}
    latex_lines.append(f"\\begin{{longtable}}{{l l *{{{expected_cols}}}{{S[table-format=2.1, table-parse-only]}}}}")
    latex_lines.append(f"\\caption{{{s_caption}}}\\label{{{s_label}}}\\\\")
    latex_lines.append("\\toprule")

    header_parts = ["Method", "Dist"]
    for s_col_name in VOCSIM_APPENDIX_S_COLUMN_ORDER:
        display_col_name = PRETTY_NAMES.get(s_col_name, s_col_name)
        if display_col_name == "Avg (Blind)":
            header_parts.append("\\multicolumn{1}{c}{\\makecell{Avg\\\\(Blind)}}")
        else:
            header_parts.append(f"\\multicolumn{{1}}{{c}}{{{sanitize(display_col_name)}}}")

    header_full_line = " & ".join(header_parts) + " \\\\"

    latex_lines.append(header_full_line)
    latex_lines.append("\\midrule")
    latex_lines.append("\\endfirsthead")
    latex_lines.append("")
    latex_lines.append(f"\\caption[]{{(Continued) {s_caption}}}\\\\")
    latex_lines.append("\\toprule")
    latex_lines.append(header_full_line)
    latex_lines.append("\\midrule")
    latex_lines.append("\\endhead")
    latex_lines.append("")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\endlastfoot")
    latex_lines.append("")

    for (method_val_idx, dist_val_idx), row_series_data in df_data.iterrows():
        row_str_parts = [sanitize(str(method_val_idx)), sanitize(str(dist_val_idx))]
        for cell_val_str in row_series_data:
            cell_content = str(cell_val_str)
            is_plain_number = False
            try:
                float(cell_content)
                is_plain_number = True
            except (ValueError, TypeError):
                is_plain_number = False

            if is_plain_number:
                row_str_parts.append(cell_content)
            else:
                row_str_parts.append(f"{{{cell_content}}}")
                
        latex_lines.append(" & ".join(row_str_parts) + " \\\\")

    latex_lines.append("\\end{longtable}")

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(latex_lines))
        logger.info(f"VocSim Appendix LaTeX table '{s_caption}' saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving VocSim Appendix LaTeX table {output_file}: {e}")