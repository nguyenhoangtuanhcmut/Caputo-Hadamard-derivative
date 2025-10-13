# -*- coding: utf-8 -*-
import io
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor, implicit_application
)
import matplotlib.pyplot as plt

# ====== Parser cấu hình (tương đương bản PyQt6) ======
TRANSFORMS = (
    standard_transformations
    + (implicit_multiplication_application, implicit_application, convert_xor)
)
LOCAL = {
    'pi': sp.pi, 'e': sp.E,
    'sqrt': sp.sqrt, 'abs': sp.Abs, 'Abs': sp.Abs,
    'ln': sp.log, 'log': sp.log,
    'log10': lambda a: sp.log(a, 10),
    'log2':  lambda a: sp.log(a, 2),
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
}

def parse_user_expr(s: str):
    """Parse KHÔNG evaluate để giữ đúng cấu trúc nhập vào (hiển thị LaTeX đầy đủ)."""
    s = s.strip().replace('÷', '/').replace('×', '*').replace('−', '-')
    s = s.replace('^', '**').replace('√(', 'sqrt(')
    return parse_expr(s, local_dict=LOCAL, transformations=TRANSFORMS, evaluate=False)

# ====== Caputo ======
def caputo_derivative(expr: sp.Expr, var: sp.Symbol, alpha: sp.Expr) -> sp.Expr:
    """
    - alpha nguyên (kể cả 0): đạo hàm thường bậc n.
    - alpha phân số dương: biểu thức dạng tích phân Caputo (không số hoá).
    """
    if alpha.is_integer is True:
        n = int(alpha)
        if n == 0:
            return sp.simplify(expr)
        return sp.diff(sp.simplify(expr), var, n)

    a0 = sp.S(0)
    t = sp.Symbol(f"{var.name}_t", real=True)
    n = int(sp.ceiling(alpha))
    integrand = sp.diff(sp.simplify(expr), var, n).subs({var: t}) * (var - t)**(n - alpha - 1)
    cap = sp.gamma(n - alpha)**-1 * sp.Integral(integrand, (t, a0, var))
    return cap

# ====== State ======
if "last_result_expr" not in st.session_state:
    st.session_state.last_result_expr: Optional[sp.Expr] = None

st.set_page_config(page_title="Mô phỏng đạo hàm Caputo-Hadamard", layout="wide")
st.title("Mô phỏng đạo hàm Caputo-Hadamard")

with st.sidebar:
    st.markdown("### Hướng dẫn nhanh")
    st.markdown(
        "- Nhập hàm \(f(x,y,z,t)\)\n"
        "- Chọn biến đạo hàm, bậc α cho Caputo-Hadamard\n"
        "- Bấm **ĐH cổ điển** hoặc **ĐH Caputo-Hadamard** để tính\n"
        "- Vẽ đồ thị: chọn biến vẽ, đoạn \([a,b]\), nhập **giá trị các biến khác** rồi bấm **Vẽ đồ thị**"
    )
    st.markdown("---")
    st.markdown("**Mẹo:** `^` cho lũy thừa; `sqrt(...)`, `ln(...)`, `sin(...)`, ...")

# ====== Nhập biểu thức & lựa chọn biến ======
col_input, col_opts = st.columns([3, 2], vertical_alignment="bottom")
with col_input:
    expr_text = st.text_input("Nhập biểu thức cần tính:", value="", placeholder="Ví dụ: sin(x)*exp(y) + x^2/(1+z^2)")
with col_opts:
    alpha_text = st.text_input("Bậc α (Caputo-Hadamard):", value="0.5", help="Ví dụ: 1/2, 1, 2, 0.7")
    var_diff_default = "x"

# Quét biến
vars_list: List[str] = ['x', 'y', 'z', 't']
parsed_expr = None
parse_error = None
if expr_text.strip():
    try:
        parsed_expr = parse_user_expr(expr_text)
        syms = sorted([s.name for s in parsed_expr.free_symbols])
        if syms:
            vars_list = syms
            var_diff_default = syms[0]
    except Exception as e:
        parse_error = str(e)

left, right = st.columns([1, 1])
with left:
    var_diff = st.selectbox("Biến đạo hàm:", options=vars_list, index=max(0, vars_list.index(var_diff_default)))
with right:
    st.markdown("&nbsp;")
    b1, b2 = st.columns(2)
    with b1:
        do_diff = st.button("Tính ĐH cổ điển", use_container_width=True)
    with b2:
        do_caputo = st.button("Tính ĐH C-H", use_container_width=True)

# Hiển thị công thức gốc
st.markdown("#### Công thức đang nhập")
if parse_error:
    st.error(f"Biểu thức không hợp lệ: {parse_error}")
elif parsed_expr is not None:
    st.latex(sp.latex(parsed_expr))
else:
    st.info("Đang đợi nhập biểu thức ...")

# ====== Kết quả ======
result_placeholder = st.empty()

def show_expr_result(title: str, expr: sp.Expr):
    with result_placeholder.container():
        st.markdown(f"### {title}")
        try:
            st.latex(sp.latex(expr))
        except Exception:
            st.code(str(expr))
    st.session_state.last_result_expr = expr

if do_diff:
    if not parsed_expr:
        st.warning("Vui lòng nhập biểu thức trước.")
    else:
        try:
            var = sp.Symbol(var_diff)
            d = sp.diff(sp.simplify(parsed_expr), var)
            show_expr_result("Kết quả đạo hàm cổ điển", d)
        except Exception as e:
            st.error(f"Không tính được đạo hàm cổ điển: {e}")

if do_caputo:
    if not parsed_expr:
        st.warning("Vui lòng nhập biểu thức trước.")
    else:
        try:
            var = sp.Symbol(var_diff)
            alpha = sp.nsimplify(alpha_text) if alpha_text.strip() else sp.Rational(1, 2)
            if alpha.is_real is False:
                st.error("α phải là số thực.")
            elif alpha.is_number and alpha < 0:
                st.error("α phải không âm (α ≥ 0).")
            else:
                cap = caputo_derivative(parsed_expr, var, alpha)
                show_expr_result(f"Đạo hàm bậc α = {sp.latex(alpha)}", cap)
        except Exception as e:
            st.error(f"Không tính được đạo hàm Caputo-Hadamard: {e}")

st.markdown("---")

# ====== Vẽ đồ thị ======
st.subheader("Vẽ đồ thị")
plot_col1, plot_col2, plot_col3 = st.columns([1, 1, 2])

with plot_col1:
    plot_var = st.selectbox("Biến vẽ:", options=vars_list, index=max(0, vars_list.index(var_diff_default)))
with plot_col2:
    a_text = st.text_input("a (trái):", "-5")
    b_text = st.text_input("b (phải):", "5")

# Chọn biểu thức để vẽ
expr_to_plot: Optional[sp.Expr] = None
if st.session_state.last_result_expr is not None and not isinstance(st.session_state.last_result_expr, sp.Integral):
    expr_to_plot = sp.simplify(st.session_state.last_result_expr)
elif parsed_expr is not None:
    expr_to_plot = sp.simplify(parsed_expr)

# Nhập giá trị các biến khác
other_vals: Dict[sp.Symbol, float] = {}
other_inputs = st.container()
with other_inputs:
    if expr_to_plot is not None:
        syms_plot = sorted([s for s in expr_to_plot.free_symbols if s.name != plot_var], key=lambda x: x.name)
        if syms_plot:
            st.markdown("**Giá trị các biến khác:**")
            cols = st.columns(min(4, len(syms_plot)) or 1)
            entered: Dict[str, str] = {}
            for i, s in enumerate(syms_plot):
                with cols[i % len(cols)]:
                    entered[str(s)] = st.text_input(f"{s} =", placeholder="nhập số…", key=f"other_{s}_{i}")
            valid = True
            for name, txt in entered.items():
                if txt.strip() == "":
                    valid = False
                else:
                    try:
                        other_vals[sp.Symbol(name)] = float(sp.N(sp.nsimplify(txt)))
                    except Exception:
                        valid = False
                        st.error(f"Giá trị của '{name}' không hợp lệ.")
            need_other_ok = (len(syms_plot) == 0) or valid
        else:
            st.caption("_(không có biến khác)_")
            need_other_ok = True
    else:
        need_other_ok = False

plot_btn = st.button("Vẽ đồ thị", type="primary")

if plot_btn:
    if expr_to_plot is None:
        st.warning("Chưa có biểu thức để vẽ.")
    else:
        # Đọc [a,b]
        try:
            a = float(sp.N(sp.nsimplify(a_text or "-5")))
            b = float(sp.N(sp.nsimplify(b_text or "5")))
        except Exception:
            st.error("Giá trị [a,b] không hợp lệ.")
            a = b = 0.0

        if not (np.isfinite(a) and np.isfinite(b)) or a >= b:
            st.error("Điều kiện: a < b")
        elif not need_other_ok:
            st.error("Vui lòng nhập đầy đủ và hợp lệ giá trị các biến khác.")
        else:
            plot_sym = sp.Symbol(plot_var)
            expr_plot = expr_to_plot.subs(other_vals) if other_vals else expr_to_plot

            try:
                f = sp.lambdify(plot_sym, expr_plot, modules=['numpy'])
            except Exception as e:
                st.error(f"Không thể chuyển biểu thức sang hàm số: {e}")
                f = None

            if f is not None:
                xs = np.linspace(a, b, 800)
                try:
                    ys = np.array(f(xs), dtype=float)
                except Exception:
                    ys_list = []
                    for xv in xs:
                        try:
                            yv = float(f(float(xv)))
                            ys_list.append(yv if np.isfinite(yv) else np.nan)
                        except Exception:
                            ys_list.append(np.nan)
                    ys = np.array(ys_list, dtype=float)

                if np.all(~np.isfinite(ys)):
                    st.info("Không có điểm hữu hiệu để vẽ trên [a,b].")
                else:
                    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
                    ax.plot(xs, ys)
                    ax.set_xlabel(f"{plot_sym}")
                    ax.set_ylabel("Giá trị của hàm số")
                    ax.set_title("Đồ thị của hàm số đang khảo sát")
                    ax.grid(True, alpha=0.1)
                    st.pyplot(fig)
