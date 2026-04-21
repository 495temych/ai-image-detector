const revealer = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      e.target.classList.add('in');
      revealer.unobserve(e.target);
    }
  }
}, { threshold: 0.12 });

document.querySelectorAll('.section, .colophon').forEach((el) => {
  el.classList.add('reveal');
  revealer.observe(el);
});
