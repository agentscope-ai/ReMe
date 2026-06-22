// Language switcher for the ReMe docs.
//
// Each language is a separate Jupyter Book served under /zh/ and /en/ with the
// same page filenames. This injects a 中文 / EN toggle into the furo sidebar
// (falling back to a floating control) that swaps the language path segment of
// the current URL.
(function () {
  "use strict";

  var path = window.location.pathname;
  var current, otherHref;

  if (path.indexOf("/en/") !== -1) {
    current = "en";
    otherHref = path.replace("/en/", "/zh/");
  } else if (path.indexOf("/zh/") !== -1) {
    current = "zh";
    otherHref = path.replace("/zh/", "/en/");
  } else {
    return; // not inside a language book (e.g. the root landing page)
  }

  function makeLink(label, lang) {
    var a = document.createElement("a");
    a.textContent = label;
    if (lang === current) {
      a.className = "active";
    } else {
      a.href = otherHref;
    }
    return a;
  }

  var wrap = document.createElement("div");
  wrap.className = "lang-switch";
  wrap.appendChild(makeLink("中文", "zh"));
  wrap.appendChild(makeLink("EN", "en"));

  function inject() {
    var brand = document.querySelector(".sidebar-brand");
    if (brand && brand.parentNode) {
      brand.parentNode.insertBefore(wrap, brand.nextSibling);
    } else {
      wrap.classList.add("lang-switch--float");
      document.body.appendChild(wrap);
    }
  }

  if (document.readyState !== "loading") {
    inject();
  } else {
    document.addEventListener("DOMContentLoaded", inject);
  }
})();
