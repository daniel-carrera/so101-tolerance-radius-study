// Loaded in <head> so the saved theme applies before the page paints.
// Dark is the default; a saved "light" choice switches it.
(function () {
  try {
    if (localStorage.getItem("theme") === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    }
  } catch (e) { /* storage blocked: stay dark */ }
})();
