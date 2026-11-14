document.querySelectorAll('[data-code-tabs]').forEach((group, index) => {
  const tabs = [...group.querySelectorAll('[role="tab"]')];
  const panels = [...group.querySelectorAll('[data-code-panel]')];
  function select(tab) {
    tabs.forEach(item => {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
    panels.forEach(panel => {
      panel.hidden = panel.dataset.codePanel !== tab.dataset.language;
    });
  }
  tabs.forEach((tab, i) => {
    const panel = panels.find(item => item.dataset.codePanel === tab.dataset.language);
    tab.id = `code-tab-${index}-${i}`;
    panel.id = `code-panel-${index}-${i}`;
    tab.setAttribute('aria-controls', panel.id);
    panel.setAttribute('aria-labelledby', tab.id);
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (i + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (i - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        select(tabs[next]);
        tabs[next].focus();
      }
    });
  });
});
