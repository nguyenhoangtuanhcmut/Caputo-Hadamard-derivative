# app.py
# -*- coding: utf-8 -*-
from typing import Dict, Optional

import streamlit as st
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor, implicit_application
)
import matplotlib.pyplot as plt
import mpmath as mp

# ====================== Parser cấu hình ======================
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
    """Parse KHÔNG evaluate để giữ cấu trúc nhập (phục vụ LaTeX)."""
    s = s.strip().replace('÷', '/').replace('×', '*').replace('−', '-')
    s = s.replace('^', '**').replace('√(', 'sqrt(')
    return parse_expr(s, local_dict=LOCAL, transformations=TRANSFORMS, evaluate=False)

def get_t_symbol(expr: sp.Expr) -> sp.Symbol:
    """Lấy đúng bản thể Symbol 't' trong biểu thức (nếu có), nếu không thì tạo Symbol('t')."""
    for s in expr.free_symbols:
        if s.name == 't':
            return s
    return sp.Symbol('t')

# ========== Định nghĩa Caputo–Hadamard (trái) theo biến t ==========
def delta_operator(expr: sp.Expr, var: sp.Symbol, k: int) -> sp.Expr:
    """Toán tử δ = t d/dt lặp k lần: (δ^k f)(t)."""
    g = sp.simplify(expr)
    for _ in range(k):
        g = sp.simplify(var * sp.diff(g, var))
    return g

def caputo_hadamard_symbolic(expr: sp.Expr, var: sp.Symbol, alpha: sp.Expr, a0: sp.Expr) -> sp.Expr:
    r"""
    {}^{CH}D^{α}_{a^+} f(t) =
      - Nếu α = n ∈ ℕ: (δ^n f)(t), δ = t d/dt.
      - Nếu n-1 < α < n: (1/Γ(n-α)) ∫_a^t (ln(t/τ))^{n-α-1} (δ^n f)(τ) dτ / τ.
    Trả về biểu thức SymPy (Integral nếu α không nguyên).
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
    integrand = sp.simplify(delta_n_f * (sp.log(var / tau))**(kernel_pow) / tau)
    return sp.gamma(n - alpha)**-1 * sp.Integral(integrand, (tau, a0, var))

# ====================== Streamlit UI ======================
st.set_page_config(page_title="Đạo hàm theo t: cổ điển & Caputo–Hadamard", layout="wide")
st.title("Đạo hàm theo biến t: cổ điển & Caputo–Hadamard")

# Sidebar: mốc trái a cho C–H
with st.sidebar:
    st.markdown("### Thiết lập C–H")
    a0_text = st.text_input("Mốc trái a (>0):", value="1", help="Ví dụ: 1, 2, E, ...")
    st.markdown("---")
    st.caption("Mọi đạo hàm đều theo **biến t**. Khi vẽ cần nhập giá trị các biến khác (khác t).")

# Nhập biểu thức
expr_text = st.text_input("Nhập biểu thức f(t, ...):", value="", placeholder="Ví dụ: exp(t) + t^2/(1+x^2)")

parsed_expr: Optional[sp.Expr] = None
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

# ===== Nút tính đạo hàm (theo t) + ô nhập α ngay trước khi tính C–H =====
col_btn = st.columns([2, 2, 2])
with col_btn[0]:
    do_diff = st.button("Tính đạo hàm cổ điển (theo t)", use_container_width=True)
with col_btn[1]:
    alpha_text = st.text_input("Bậc α cho C–H:", value="", key="alpha_input",
                               placeholder="vd: 0.5 hoặc 2")
with col_btn[2]:
    do_ch = st.button("Tính đạo hàm Caputo–Hadamard", use_container_width=True)

result_placeholder = st.empty()
if "last_classic" not in st.session_state:
    st.session_state.last_classic: Optional[sp.Expr] = None
if "last_ch" not in st.session_state:
    st.session_state.last_ch: Optional[sp.Expr] = None

def show_expr_result(title: str, expr: sp.Expr, key_store: str):
    with result_placeholder.container():
        st.markdown(f"### {title}")
        try:
            st.latex(sp.latex(expr))
        except Exception:
            st.code(str(expr))
    st.session_state[key_store] = expr

# --- Tính đạo hàm cổ điển ---
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

# --- Tính Caputo–Hadamard ---
if do_ch:
    if not parsed_expr:
        st.warning("Vui lòng nhập biểu thức trước.")
    elif alpha_text.strip() == "":
        st.warning("Vui lòng nhập bậc đạo hàm α trước khi tính C–H.")
    else:
        try:
            t = get_t_symbol(parsed_expr)
            alpha = sp.nsimplify(alpha_text)
            a0 = sp.nsimplify(a0_text) if a0_text.strip() else sp.S(1)
            ch = caputo_hadamard_symbolic(parsed_expr, t, alpha, a0)
            show_expr_result(
                f"CH-f'(t) – Caputo–Hadamard bậc α = {sp.latex(alpha)}, mốc a = {sp.latex(a0)}",
                ch, "last_ch"
            )
        except Exception as e:
            st.error(f"Không tính được Caputo–Hadamard: {e}")

st.markdown("---")

# ====================== Vẽ đồ thị ======================
st.subheader("Vẽ đồ thị theo biến t")
plot_target = st.radio(
    "Chọn đối tượng để vẽ",
    options=["f", "f'", "CH-f'"],
    index=0,
    horizontal=True,
)

col_ab = st.columns(2)
with col_ab[0]:
    a_text = st.text_input("a (trái):", "-5")
with col_ab[1]:
    b_text = st.text_input("b (phải):", "5")

# --------- Xác định biểu thức cơ sở để lấy biến phụ & để vẽ ---------
expr_base: Optional[sp.Expr] = None
expr_to_plot: Optional[sp.Expr] = None
source_note = ""

if plot_target == "f":
    expr_base = parsed_expr
    expr_to_plot = sp.simplify(parsed_expr) if parsed_expr is not None else None
    source_note = "Đang vẽ: f(t)"
elif plot_target == "f'":
    expr_base = parsed_expr
    if st.session_state.get("last_classic") is not None:
        expr_to_plot = sp.simplify(st.session_state.last_classic)
    elif parsed_expr is not None:
        t = get_t_symbol(parsed_expr)
        expr_to_plot = sp.diff(sp.simplify(parsed_expr), t)
    source_note = "Đang vẽ: f'(t) – đạo hàm cổ điển"
else:  # "CH-f'"
    expr_base = parsed_expr  # các biến phụ lấy từ f
    ch_expr = st.session_state.get("last_ch")
    if ch_expr is None and parsed_expr is not None:
        # Tính nhanh nếu chưa bấm nút
        if alpha_text.strip() == "":
            st.warning("Vui lòng nhập bậc α ở ô trên để vẽ CH-f'.")
        else:
            try:
                t = get_t_symbol(parsed_expr)
                alpha_tmp = sp.nsimplify(alpha_text)
                a0_tmp = sp.nsimplify(a0_text) if a0_text.strip() else sp.S(1)
                ch_expr = caputo_hadamard_symbolic(parsed_expr, t, alpha_tmp, a0_tmp)
                st.session_state.last_ch = ch_expr
            except Exception as e:
                st.error(f"Không tạo được CH-f' để vẽ: {e}")
                ch_expr = None
    expr_to_plot = ch_expr
    source_note = "Đang vẽ: CH-f'(t)"

if source_note:
    st.caption(source_note)

# --------- Nhập giá trị các biến khác (khác t) ---------
other_vals: Dict[sp.Symbol, float] = {}
need_other_ok = False
if expr_base is not None:
    t_var = get_t_symbol(expr_base)
    syms_plot = sorted([s for s in expr_base.free_symbols if s != t_var], key=lambda x: x.name)
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

# --------- Helpers vẽ an toàn ---------
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def _eval_curve_numpy(f, xs: np.ndarray) -> np.ndarray:
    """Chuẩn hoá đầu ra của f(xs) -> mảng 1D cùng shape với xs, lọc complex/NaN."""
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
    if expr_base is None or expr_to_plot is None:
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
            t = get_t_symbol(expr_base)

            # --------- Trường hợp f hoặc f' (hoặc CH-f' nhưng α nguyên) -> lambdify ---------
            if not isinstance(expr_to_plot, sp.Integral):
                expr_plot = sp.simplify(expr_to_plot.subs(other_vals)) if other_vals else sp.simplify(expr_to_plot)
                try:
                    f = sp.lambdify(t, expr_plot, modules=['numpy'])
                except Exception as e:
                    st.error(f"Không thể chuyển biểu thức sang hàm số: {e}")
                    f = None

                if f is not None:
                    xs = np.linspace(a, b, 600)
                    ys = _eval_curve_numpy(f, xs)
                    if ys.shape != xs.shape:
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

            # --------- Trường hợp CH-f' với α không nguyên -> tích phân số mpmath ---------
            else:
                # Cần alpha và a0 hợp lệ
                if alpha_text.strip() == "":
                    st.error("Vui lòng nhập bậc α để vẽ CH-f'.")
                else:
                    try:
                        alpha_val = float(sp.N(sp.nsimplify(alpha_text)))
                        a0_val = float(sp.N(sp.nsimplify(a0_text))) if a0_text.strip() else 1.0
                    except Exception:
                        st.error("α hoặc a không hợp lệ.")
                        alpha_val = None

                    if alpha_val is not None:
                        if a0_val <= 0:
                            st.error("Mốc trái a phải > 0.")
                        else:
                            n = int(np.ceil(alpha_val))  # bậc trên của delta
                            # Chuẩn bị delta^n f(tau), đã thay biến phụ
                            tau = sp.Symbol(f"{t.name}_tau", real=True, positive=True)
                            base_expr = parsed_expr.subs(other_vals) if (parsed_expr is not None and other_vals) else parsed_expr
                            if base_expr is None:
                                st.error("Thiếu biểu thức cơ sở để vẽ.")
                            else:
                                delta_n = delta_operator(sp.simplify(base_expr), t, n).subs({t: tau})
                                # lambdify bằng mpmath (để dùng trong quad)
                                try:
                                    f_tau = sp.lambdify(tau, delta_n, modules=['mpmath'])
                                except Exception as e:
                                    st.error(f"Không thể chuyển δ^n f(tau) sang hàm số: {e}")
                                    f_tau = None

                                if f_tau is not None:
                                    gamma_factor = 1.0 / mp.gamma(n - alpha_val)

                                    def ch_numeric(tt):
                                        # Giá trị định nghĩa: chỉ có nghĩa khi tt > a0
                                        if tt <= a0_val:
                                            return mp.nan
                                        def integrand(ta):
                                            return (mp.log(tt / ta) ** (n - alpha_val - 1.0)) * f_tau(ta) / ta
                                        try:
                                            val = mp.quad(integrand, [a0_val, tt])
                                            return gamma_factor * val
                                        except Exception:
                                            return mp.nan

                                    # Bảo đảm miền vẽ nằm trên (a0, b]
                                    left = max(a, a0_val + 1e-9)
                                    if left >= b:
                                        st.error("Khoảng vẽ phải thỏa a < b và b > a0.")
                                    else:
                                        xs = np.linspace(left, b, 350)
                                        ys_list = []
                                        for xv in xs:
                                            try:
                                                yv = float(ch_numeric(float(xv)))
                                                ys_list.append(yv if np.isfinite(yv) else np.nan)
                                            except Exception:
                                                ys_list.append(np.nan)
                                        ys = np.array(ys_list, dtype=float)

                                        if np.all(~np.isfinite(ys)):
                                            st.info("Không có điểm hữu hiệu để vẽ trên khoảng đã chọn (có thể do α, a hoặc biểu thức).")
                                        else:
                                            fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
                                            ax.plot(xs, ys)
                                            ax.set_xlabel("t")
                                            ax.set_ylabel("giá trị xấp xỉ")
                                            ax.set_title(f"Đồ thị: CH-f'(t) (xấp xỉ số, α={alpha_val})")
                                            ax.grid(True, alpha=0.2)
                                            st.pyplot(fig)
