// Keep the established unlayered CSS cascade when compiling Tailwind 4.
// Existing application selectors and utility classes then retain their v3 priority.
module.exports = () => ({
  postcssPlugin: "mini-openclaw-preserve-css-cascade",
  OnceExit(root) {
    root.walkAtRules("layer", (rule) => {
      if (rule.nodes) rule.replaceWith(rule.nodes);
      else rule.remove();
    });
    root.walkRules((rule) => {
      // v3 put spacing and dividers on following visible siblings. Preserve that
      // behavior so existing child margins, hidden rows, and borders still work.
      const siblings = rule.selector.match(/^:where\((\.(?:space-[xy]-|divide-)[^ ]+) > :not\(:last-child\)\)$/);
      if (siblings) {
        rule.selector = `${siblings[1]} > :not([hidden]) ~ :not([hidden])`;
        for (const [first, last, firstV3, lastV3] of [
          ["margin-block-start", "margin-block-end", "margin-top", "margin-bottom"],
          ["margin-inline-start", "margin-inline-end", "margin-left", "margin-right"],
          ["border-top-width", "border-bottom-width", "border-top-width", "border-bottom-width"],
          ["border-left-width", "border-right-width", "border-left-width", "border-right-width"],
        ]) {
          const start = rule.nodes.find((node) => node.prop === first);
          const end = rule.nodes.find((node) => node.prop === last);
          if (start && end) {
            [start.value, end.value] = [end.value, start.value];
            start.prop = firstV3;
            end.prop = lastV3;
          }
        }
      }
      // Tailwind 4 adds the built-in outline-none after custom utilities.
      // Remove its style override to retain the accessible v3 invisible outline.
      if (rule.selector.includes("outline-none")) {
        rule.walkDecls((declaration) => {
          if (["outline-style", "--tw-outline-style"].includes(declaration.prop)) declaration.remove();
        });
        if (!rule.nodes.some((node) => node.prop === "outline")) {
          rule.append({ prop: "outline", value: "2px solid transparent" });
          rule.append({ prop: "outline-offset", value: "2px" });
        }
      }
    });
  },
});

module.exports.postcss = true;
