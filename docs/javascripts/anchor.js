// Links to a section (findings/#future-work) land before the figures above it have loaded; as they load, the page
// grows and the section slides down. Once everything has loaded, scroll to the linked section again.
window.addEventListener("load", () => {
  if (!location.hash) return;
  const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (target) target.scrollIntoView();
});
