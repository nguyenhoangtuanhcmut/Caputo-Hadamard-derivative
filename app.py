# app.py
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

# ====== Parser cấu hình ======
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

# ====== Định nghĩa Caputo–Hadamard ======
def delta_operator(expr: sp.Expr, var: sp.Symbol, k: int) -> sp.Expr:
    """
    Toán tử δ = t d/dt lặp k lần: (δ^k f)(t) với var đóng vai trò 't'.
    """
    g = sp.simplify(expr)
    for _ in range(k):
        g = sp.simplify(var * sp.diff(g, var))
    return g

def caputo_hadamard_derivative(expr: sp.Expr, var: sp.Symbol, alpha: sp.Expr, a0: sp.Expr) -> sp.Expr:
    r"""
    Đạo hàm Caputo–Hadamard trái bậc alpha trên (a0, ·):
    - Nếu alpha = n ∈ ℕ: (δ^n f)(x) với δ = x d/dx.
    - Nếu alpha = 0: f(x).
    - Nếu n-1 < alpha < n:
        1/Γ(n-α) ∫_a0^x (ln(x/t))^{n-α-1} (δ^n f)(t) dt / t.
    """
    alpha = sp.nsimplify(alpha)
    a0 = sp.nsimplify(a0)

    if (alpha.is_real is False) or (alpha.is_number and alpha < 0):
        raise ValueError("α phải là số thực và không âm.")
    if a0.is_number and (a0 <= 0):
        raise ValueError("Mốc trái a phải > 0 để ln(x/t) có nghĩa.")

    # Trường hợp nguyên
    if alpha.is_integer is True:
        n = int(alpha)
        if n == 0:
            return sp.simplify(expr)
        return delta_operator(expr, var, n)

    # Trường hợp không nguyên
    n = int(sp.ceiling(alpha))
    t = sp.Symbol(f"{var.name}_t", real=True, positive=True)
    delta_n_f = delta_operator(sp.simplify(expr), var, n).subs({var: t})
    kernel_pow = n - alpha - 1
    integrand = delta_n_f * (sp.log(var / t))**(kernel_pow) / t
    return sp.gamma(n - alpha)**-1 * sp.Integral(sp.simplify(integrand), (t, a0, var))

# ====== State ======
if "last_result_expr" not in st.session_state:
    st.session_state.last_result_expr: Optional[sp.Expr] = None

st.set_page_config(page_title="Mô phỏng đạo hàm Caputo–Hadamard", layout="wide")
st.title("Mô phỏng đạo hàm Caputo–Hadamard")

# ====== Sidebar ======
with st.sidebar:
    st.markdown("### Hướng dẫn nhanh")
    st.markdown(
        "- Nhập hàm \(f(x,y,z,t)\)\n"
        "- Chọn biến đạo hàm, bậc α và **mốc trái a>0** cho Caputo–Hadamard\n"
        "- Bấm **Tính ĐH C–H** để hiển thị kết quả (dạng tích phân khi α không nguyên)\n"
        "- Vẽ đồ thị: chọn biến vẽ, đoạn \([a,b]\), nhập **giá trị các biến khác** rồi bấm **Vẽ đồ thị**"
    )
    st.markdown("---")
    st.markdown("**Mẹo nhập:** `^` (lũy thừa); `sqrt(...)`, `ln(...)`, `sin(...)`, ...")
    st.markdown("---")
    a0_text = st.text_input("Mốc trái a (>0) cho C–H:", value="1", help="Ví dụ: 1, 2, E, ...")

# ====== Nhập biểu thức & lựa chọn biến ======
col_input, col_opts = st.columns([3, 2], vertical_alignment="bottom")
with col_input:
    expr_text = st.text_input("Nhập biểu thức cần tính:", value="", placeholder="Ví dụ: sin(x)*exp(y) + x^2/(1+z^2)")
with col_opts:
    alpha_text = st.text_input("Bậc α (Caputo–Hadamard):", value="0.5", help="Ví dụ: 1/2, 1, 2, 0.7")
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
    b2, = st.columns(1)
    with b2:
        do_caputo = st.button("Tính ĐH C–H", use_container_width=True)

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

if do_caputo:
    if not parsed_expr:
        st.warning("Vui lòng nhập biểu thức trước.")
    else:
        try:
            # var dương để log(x/t) có nghĩa khi x>0
            var = sp.Symbol(var_diff, real=True, positive=True)
            alpha = sp.nsimplify(alpha_text) if alpha_text.strip() else sp.Rational(1, 2)
            a0 = sp.nsimplify(a0_text) if a0_text.strip() else sp.S(1)

            cap = caputo_hadamard_derivative(parsed_expr, var, alpha, a0)
            show_expr_result(
                f"Kết quả đạo hàm Caputo-Hadamard theo bậc α = {sp.latex(alpha)}, mốc a = {sp.latex(a0)}",
                cap
            )
        except Exception as e:
            st.error(f"Không tính được Caputo–Hadamard: {e}")

st.markdown("---")

# ====== Vẽ đồ thị ======
st.subheader("Vẽ đồ thị")
plot_col1, plot_col2, plot_col3 = st.columns([1, 1, 2])

with plot_col1:
    plot_var = st.selectbox("Biến vẽ:", options=vars_list, index=max(0, vars_list.index(var_diff_default)))
with plot_col2:
    a_text = st.text_input("a (trái):", "-5")
    b_text = st.text_input("b (phải):", "5")

# Chọn biểu thức để vẽ: ưu tiên kết quả gần nhất nếu KHÔNG phải Integral (CH không nguyên là Integral)
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

def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def _eval_curve(f, xs: np.ndarray) -> np.ndarray:
    """
    Chuẩn hoá đầu ra của f(xs) để luôn thành mảng 1D cùng shape với xs.
    - Scalar -> broadcast
    - (N,1)/(1,N) -> ravel về (N,)
    - Complex nhỏ phần ảo -> lấy phần thực; ngược lại -> NaN
    - Lọc giá trị không hữu hạn
    """
    try:
        y = f(xs)
    except Exception:
        # fallback: đánh giá từng điểm
        y = [_safe_float(f(float(x))) for x in xs]

    y = np.asarray(y)

    # Scalar -> broadcast
    if y.ndim == 0:
        y = np.full_like(xs, _safe_float(y))

    # (N,1) hoặc (1,N) -> (N,)
    if y.ndim > 1:
        y = np.ravel(y)

    # ép float
    y = y.astype(float, copy=False)

    # Complex -> xử lý
    if np.iscomplexobj(y):
        imag = np.abs(np.imag(y))
        y = np.where(imag < 1e-12, np.real(y), np.nan)

    # Lọc NaN/Inf
    y[~np.isfinite(y)] = np.nan
    return y

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
            st.error("Điều kiện: a < b và là số hữu hạn.")
        elif not need_other_ok:
            st.error("Vui lòng nhập đầy đủ và hợp lệ giá trị các biến khác.")
        else:
            plot_sym = sp.Symbol(plot_var, real=True)

            expr_plot = expr_to_plot.subs(other_vals) if other_vals else expr_to_plot

            # Tạo hàm số và vẽ
            try:
                f = sp.lambdify(plot_sym, expr_plot, modules=['numpy'])
            except Exception as e:
                st.error(f"Không thể chuyển biểu thức sang hàm số: {e}")
                f = None

            if f is not None:
                xs = np.linspace(a, b, 800)
                ys = _eval_curve(f, xs)

                # Phòng xa: ép cùng shape
                if ys.shape != xs.shape:
                    ys = np.resize(ys, xs.shape)

                if np.all(~np.isfinite(ys)):
                    st.info("Không có điểm hữu hiệu để vẽ trên [a,b].")
                else:
                    # Nếu hàm hằng, thông báo nhẹ
                    if float(np.nanstd(ys)) == 0.0:
                        st.caption("Hàm (sau khi thay biến khác) là **hằng số** theo biến vẽ.")

                    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
                    ax.plot(xs, ys)
                    ax.set_xlabel(f"{plot_sym}")
                    ax.set_ylabel("Giá trị của hàm số")
                    ax.set_title("Đồ thị của hàm số đang khảo sát")
                    ax.grid(True, alpha=0.2)
                    st.pyplot(fig)
