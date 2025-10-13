# app.py
# -*- coding: utf-8 -*-
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
    'exp': sp.exp,  # tiện nhập exp(t)
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
}

def parse_user_expr(s: str):
    """Parse KHÔNG evaluate để giữ đúng cấu trúc nhập vào (hiển thị LaTeX đầy đủ)."""
    s = s.strip().replace('÷', '/').replace('×', '*').replace('−', '-')
    s = s.replace('^', '**').replace('√(', 'sqrt(')
    return parse_expr(s, local_dict=LOCAL, transformations=TRANSFORMS, evaluate=False)

def get_t_symbol(expr: sp.Expr) -> sp.Symbol:
    """Lấy đúng bản thể Symbol 't' trong biểu thức (nếu có), nếu không thì tạo Symbol('t')."""
    for s in expr.free_symbols:
        if s.name == 't':
            return s
    return sp.Symbol('t')

# ====== Định nghĩa Caputo–Hadamard (trái) theo biến t ======
def delta_operator(expr: sp.Expr, var: sp.Symbol, k: int) -> sp.Expr:
    """Toán tử δ = t d/dt lặp k lần: (δ^k f)(t)."""
    g = sp.simplify(expr)
    for _ in range(k):
        g = sp.simplify(var * sp.diff(g, var))
    return g

def caputo_hadamard_derivative(expr: sp.Expr, var: sp.Symbol, alpha: sp.Expr, a0: sp.Expr) -> sp.Expr:
    r"""
    {}^{CH}D^{α}_{a^+} f(t) =
      - Nếu α = n ∈ ℕ: (δ^n f)(t), δ = t d/dt.
      - Nếu n-1 < α < n: (1/Γ(n-α)) ∫_a^t (ln(t/τ))^{n-α-1} (δ^n f)(τ) dτ / τ.
    """
    alpha = sp.nsimplify(alpha)
    a0 = sp.nsimplify(a0)

    if (alpha.is_real is False) or (alpha.is_number and alpha < 0):
        raise ValueError("α phải là số thực và không âm.")
    if a0.is_number and (a0 <= 0):
        raise ValueError("Mốc trái a phải > 0 để ln(t/τ) có nghĩa.")

    if alpha.is_integer is True:
        n = int(alpha)
        if n == 0:
            return sp.simplify(expr)
        return delta_operator(expr, var, n)

    n = int(sp.ceiling(alpha))
    tau = sp.Symbol(f"{var.name}_tau", real=True, positive=True)
    delta_n_f = delta_operator(sp.simplify(expr), var, n).subs({var: tau})
    kernel_pow = n - alpha - 1
    integrand = delta_n_f * (sp.log(var / tau))**(kernel_pow) / tau
    return sp.gamma(n - alpha)**-1 * sp.Integral(sp.simplify(integrand), (tau, a0, var))

# ====== State ======
st.set_page_config(page_title="Mô phỏng đạo hàm theo t: cổ điển & Caputo–Hadamard", layout="wide")
if "last_classic" not in st.session_state:
    st.session_state.last_classic: Optional[sp.Expr] = None
if "last_ch" not in st.session_state:
    st.session_state.last_ch: Optional[sp.Expr] = None

st.title("Mô phỏng đạo hàm theo biến t: cổ điển & Caputo–Hadamard")

# ====== Sidebar ======
with st.sidebar:
    st.markdown("### Thiết lập")
    alpha_text = st.text_input("Bậc α (cho C–H):", value="0.5", help="Ví dụ: 1/2, 1, 2, 0.7")
    a0_text = st.text_input("Mốc trái a (>0) cho C–H:", value="1", help="Ví dụ: 1, 2, E, ...")
    st.markdown("---")
    st.caption("Lưu ý: Mọi đạo hàm đều theo **biến t**. Khi vẽ, cần nhập giá trị các biến khác (nếu có).")

# ====== Nhập biểu thức ======
expr_text = st.text_input("Nhập biểu thức f(t, ...):", value="", placeholder="Ví dụ: exp(t) + t^2/(1+x^2)")
parsed_expr = None
parse_error = None
if expr_text.strip():
    try:
        parsed_expr = parse_user_expr(expr_text)
    except Exception as e:
        parse_error = str(e)

st.markdown("#### Công thức đang nhập (LaTeX)")
if parse_error:
    st.error(f"Biểu thức không hợp lệ: {parse_error}")
elif parsed_expr is not None:
    st.latex(sp.latex(parsed_expr))
else:
    st.info("Đang đợi nhập biểu thức ...")

# ====== Nút tính đạo hàm (theo t) ======
col_btn = st.columns(2)
with col_btn[0]:
    do_diff = st.button("Tính đạo hàm cổ điển (theo t)", use_container_width=True)
with col_btn[1]:
    do_ch = st.button("Tính đạo hàm Caputo–Hadamard (theo t)", use_container_width=True)

result_placeholder = st.empty()

def show_expr_result(title: str, expr: sp.Expr, key_store: str):
    with result_placeholder.container():
        st.markdown(f"### {title}")
        try:
            st.latex(sp.latex(expr))
        except Exception:
            st.code(str(expr))
    st.session_state[key_store] = expr

# === Xử lý tính toán ===
if do_diff:
    if not parsed_expr:
        st.warning("Vui lòng nhập biểu thức trước.")
    else:
        try:
            t = get_t_symbol(parsed_expr)
            if t not in parsed_expr.free_symbols:
                st.warning("Biểu thức không phụ thuộc t; đạo hàm theo t bằng 0.")
            d = sp.diff(sp.simplify(parsed_expr), t)
            show_expr_result("f'(t) – đạo hàm cổ điển theo t", d, "last_classic")
        except Exception as e:
            st.error(f"Không tính được đạo hàm cổ điển: {e}")

if do_ch:
    if not parsed_expr:
        st.warning("Vui lòng nhập biểu thức trước.")
    else:
        try:
            t = get_t_symbol(parsed_expr)  # dùng đúng biến t trong biểu thức
            alpha = sp.nsimplify(alpha_text) if alpha_text.strip() else sp.Rational(1, 2)
            a0 = sp.nsimplify(a0_text) if a0_text.strip() else sp.S(1)
            ch = caputo_hadamard_derivative(parsed_expr, t, alpha, a0)
            show_expr_result(
                f"CH-f'(t) – đạo hàm Caputo–Hadamard bậc α = {sp.latex(alpha)}, mốc a = {sp.latex(a0)}",
                ch, "last_ch"
            )
        except Exception as e:
            st.error(f"Không tính được Caputo–Hadamard: {e}")

st.markdown("---")

# ====== Vẽ đồ thị ======
st.subheader("Vẽ đồ thị theo biến t")

# Tick chọn đối tượng vẽ (mutually exclusive -> radio)
plot_target = st.radio(
    "Chọn đối tượng để vẽ",
    options=["f", "f'", "CH-f'"],
    index=0,
    horizontal=True,
)

# Chọn đoạn [a,b]
col_ab = st.columns(2)
with col_ab[0]:
    a_text = st.text_input("a (trái):", "-5")
with col_ab[1]:
    b_text = st.text_input("b (phải):", "5")

# Xác định biểu thức gốc để vẽ (theo lựa chọn)
expr_to_plot: Optional[sp.Expr] = None
source_note = ""
if plot_target == "f":
    expr_to_plot = sp.simplify(parsed_expr) if parsed_expr is not None else None
    source_note = "Đang vẽ: f(t)"
elif plot_target == "f'":
    # Ưu tiên dùng kết quả đã tính; nếu chưa, tính on-the-fly
    if st.session_state.get("last_classic") is not None:
        expr_to_plot = sp.simplify(st.session_state.last_classic)
    elif parsed_expr is not None:
        t = get_t_symbol(parsed_expr)
        expr_to_plot = sp.diff(sp.simplify(parsed_expr), t)
    source_note = "Đang vẽ: f'(t) (đạo hàm cổ điển)"
else:  # "CH-f'"
    ch_expr = st.session_state.get("last_ch")
    if ch_expr is None and parsed_expr is not None:
        # Tính on-the-fly nếu người dùng chưa bấm nút
        t = get_t_symbol(parsed_expr)
        alpha = sp.nsimplify(alpha_text) if alpha_text.strip() else sp.Rational(1, 2)
        a0 = sp.nsimplify(a0_text) if a0_text.strip() else sp.S(1)
        ch_expr = caputo_hadamard_derivative(parsed_expr, t, alpha, a0)
        st.session_state.last_ch = ch_expr

    if isinstance(ch_expr, sp.Integral):
        expr_to_plot = None
        st.info("CH-f' là **biểu thức tích phân** (α không nguyên) nên không vẽ trực tiếp. "
                "Hãy chọn α nguyên để có δ^n f(t) và vẽ được.")
    else:
        expr_to_plot = sp.simplify(ch_expr) if ch_expr is not None else None
    source_note = "Đang vẽ: CH-f'(t)"

if source_note:
    st.caption(source_note)

# Nhập giá trị cho các biến khác (khác t)
other_vals: Dict[sp.Symbol, float] = {}
if expr_to_plot is not None:
    t_var = get_t_symbol(expr_to_plot)
    syms_plot = sorted([s for s in expr_to_plot.free_symbols if s != t_var], key=lambda x: x.name)

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
        st.caption("_(không có biến khác ngoài t)_")
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
        y = [_safe_float(f(float(x))) for x in xs]

    y = np.asarray(y)
    if y.ndim == 0:
        y = np.full_like(xs, _safe_float(y))
    if y.ndim > 1:
        y = np.ravel(y)
    y = y.astype(float, copy=False)
    if np.iscomplexobj(y):
        imag = np.abs(np.imag(y))
        y = np.where(imag < 1e-12, np.real(y), np.nan)
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
            t = get_t_symbol(expr_to_plot)
            expr_plot = expr_to_plot.subs(other_vals) if other_vals else expr_to_plot

            try:
                f = sp.lambdify(t, expr_plot, modules=['numpy'])
            except Exception as e:
                st.error(f"Không thể chuyển biểu thức sang hàm số: {e}")
                f = None

            if f is not None:
                xs = np.linspace(a, b, 800)
                ys = _eval_curve(f, xs)

                if ys.shape != xs.shape:  # phòng xa
                    ys = np.resize(ys, xs.shape)

                if np.all(~np.isfinite(ys)):
                    st.info("Không có điểm hữu hiệu để vẽ trên [a,b].")
                else:
                    if float(np.nanstd(ys)) == 0.0:
                        st.caption("Hàm (sau khi thay biến khác) là **hằng số** theo t.")

                    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
                    ax.plot(xs, ys)
                    ax.set_xlabel("t")
                    ax.set_ylabel("giá trị")
                    ax.set_title(f"Đồ thị: {plot_target}")
                    ax.grid(True, alpha=0.2)
                    st.pyplot(fig)
