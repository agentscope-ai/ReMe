/** Styles for the Environment tab: the hero, the step cards, and the .env form. */
export const environmentStyles = `
.reme-env{display:flex;flex-direction:column;gap:12px;padding:16px 0 8px}
.reme-env-hero{position:relative;display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:14px 18px;overflow:hidden;border:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 18%,var(--dsw-alias-border-l2));border-radius:16px;background:linear-gradient(128deg,color-mix(in srgb,var(--dsw-alias-brand-primary) 8%,var(--dsw-alias-bg-layer-3)),var(--dsw-alias-bg-layer-3) 58%);padding:17px 19px}
.reme-env-hero:after{position:absolute;right:-110px;bottom:-170px;width:230px;height:230px;border-radius:50%;background:color-mix(in srgb,var(--dsw-alias-brand-primary) 7%,transparent);content:"";pointer-events:none}
.reme-env-hero-brand{display:flex;min-width:0;align-items:center;gap:12px}
.reme-env-hero-mark{display:grid;width:40px;height:40px;flex:none;place-items:center;border-radius:12px;background:var(--dsw-alias-label-primary);color:var(--dsw-alias-bg-layer-3);font-size:13px;font-weight:750;letter-spacing:-.4px;box-shadow:0 8px 22px color-mix(in srgb,var(--dsw-alias-label-primary) 12%,transparent)}
.reme-env-hero-copy{min-width:0}
.reme-env-eyebrow{display:block;color:var(--dsw-alias-label-tertiary);font-size:10px;font-weight:650;letter-spacing:.08em;line-height:1.4;text-transform:uppercase}
.reme-env-hero-copy h3{margin:2px 0 0;font-size:17px;line-height:1.35}
.reme-env-hero-sub{margin:3px 0 0;overflow:hidden;color:var(--dsw-alias-label-tertiary);font-size:11px;line-height:1.4;text-overflow:ellipsis;white-space:nowrap}
.reme-env-hero-chips{position:relative;z-index:1;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px}
.reme-env-hero-actions{grid-column:1 / -1;display:flex;flex-wrap:wrap;align-items:center;gap:8px;border-top:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 14%,var(--dsw-alias-border-l2));padding-top:12px}
.reme-env-hero-meta{margin-left:auto;color:var(--dsw-alias-label-tertiary);font-size:11px}
.reme-env-hero-error,.reme-env-notice{margin:0}
.reme-env-card{border:1px solid var(--dsw-alias-border-l2);border-radius:13px;background:var(--dsw-alias-bg-layer-3)}
.reme-env-card>header{display:flex;align-items:flex-start;gap:11px;padding:13px 15px}
.reme-env-card.collapsed>header{padding-bottom:14px}
.reme-env-step{display:grid;width:22px;height:22px;flex:none;place-items:center;margin-top:-1px;border-radius:50%;background:color-mix(in srgb,var(--dsw-alias-brand-primary) 12%,transparent);color:var(--dsw-alias-brand-primary);font-size:11px;font-weight:700;font-variant-numeric:tabular-nums}
.reme-env-card>header>div{min-width:0;flex:1}
.reme-env-card>header h4{margin:0;font-size:13px;font-weight:600;line-height:1.5}
.reme-env-card>header p{display:-webkit-box;margin:2px 0 0;overflow:hidden;color:var(--dsw-alias-label-tertiary);font-size:11px;line-height:1.5;-webkit-box-orient:vertical;-webkit-line-clamp:2}
.reme-env-card>header p.reme-env-summary{display:block;text-overflow:ellipsis;white-space:nowrap}
.reme-env-card>header aside{display:flex;flex:none;align-items:center;gap:8px}
.reme-env-count{flex:none;border-radius:999px;background:var(--dsw-alias-bg-module-platform);padding:2px 9px;color:var(--dsw-alias-label-secondary);font-size:11px;line-height:17px;white-space:nowrap}
.reme-env-card-body{display:flex;flex-direction:column;gap:10px;border-top:1px solid var(--dsw-alias-border-l2);padding:12px 15px 14px}
.reme-env-picked{display:flex;align-items:center;gap:11px;border:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 30%,var(--dsw-alias-border-l2));border-radius:11px;background:color-mix(in srgb,var(--dsw-alias-brand-primary) 5%,var(--dsw-alias-bg-layer-3));padding:10px 12px}
.reme-env-picked-mark{width:16px;height:16px;flex:none;box-sizing:border-box;border:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 45%,transparent);border-radius:50%;background:var(--dsw-alias-brand-primary);background-clip:content-box;padding:4px}
.reme-env-picked-main{display:flex;min-width:0;flex:1;flex-direction:column;gap:3px}
.reme-env-title{display:flex;min-width:0;flex-wrap:wrap;align-items:center;gap:6px}
.reme-env-title strong{font-size:13px;font-weight:600;line-height:1.5}
.reme-env-tag{flex:none;border:1px solid var(--dsw-alias-border-l2);border-radius:999px;padding:1px 7px;color:var(--dsw-alias-label-secondary);font-size:10.5px;line-height:15px;white-space:nowrap}
.reme-env-tag.ok{border-color:color-mix(in srgb,#21a366 25%,transparent);background:color-mix(in srgb,#21a366 8%,transparent);color:#168653}
.reme-env-tag.brand{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 28%,transparent);background:color-mix(in srgb,var(--dsw-alias-brand-primary) 8%,transparent);color:var(--dsw-alias-brand-primary)}
.reme-env-tag.warn{border-color:color-mix(in srgb,#d99416 30%,transparent);background:color-mix(in srgb,#d99416 9%,transparent);color:color-mix(in srgb,#d99416 72%,var(--dsw-alias-label-primary))}
.reme-env-tag.bad{border-color:color-mix(in srgb,var(--dsw-alias-label-error) 25%,transparent);background:color-mix(in srgb,var(--dsw-alias-label-error) 7%,transparent);color:var(--dsw-alias-label-error)}
.reme-env-path{min-width:0;overflow:hidden;color:var(--dsw-alias-label-tertiary);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;line-height:1.5;text-overflow:ellipsis;white-space:nowrap}
.reme-env-note{margin:0}
.reme-env-picker{display:flex;flex-direction:column;gap:9px}
.reme-env-picker-bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.reme-env-picker-bar .reme-settings-muted{margin:0}
.reme-env-search{box-sizing:border-box;flex:1;min-width:180px;border:1px solid var(--dsw-alias-border-l2);border-radius:9px;background:var(--dsw-alias-bg-layer-3);padding:6px 10px;color:var(--dsw-alias-label-primary);font:inherit;font-size:12px;line-height:1.5}
.reme-env-search:focus-visible{border-color:var(--dsw-alias-brand-primary);outline:0}
.reme-env-options{display:flex;flex-direction:column;overflow:hidden;border:1px solid var(--dsw-alias-border-l2);border-radius:11px;background:var(--dsw-alias-bg-layer-3)}
.reme-env-option{display:flex;align-items:center;gap:8px;padding:8px 11px;cursor:pointer}
.reme-env-option+.reme-env-option{border-top:1px solid var(--dsw-alias-border-l2)}
.reme-env-option:hover{background:color-mix(in srgb,var(--dsw-alias-label-primary) 3%,transparent)}
.reme-env-option input{flex:none;margin:0;accent-color:var(--dsw-alias-brand-primary)}
.reme-env-option strong{flex:none;font-size:12px;font-weight:600;line-height:1.5;white-space:nowrap}
.reme-env-option em{flex:none;color:var(--dsw-alias-label-tertiary);font-size:11px;font-style:normal;line-height:1.5;white-space:nowrap}
.reme-env-option-path{flex:1;min-width:60px;overflow:hidden;text-align:right;color:var(--dsw-alias-label-tertiary);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:10.5px;line-height:1.5;text-overflow:ellipsis;white-space:nowrap}
.reme-env-more{align-self:flex-start;color:var(--dsw-alias-brand-primary);font-size:12px}
.reme-env-hint{display:flex;flex-direction:column;align-items:flex-start;gap:8px;border:1px dashed var(--dsw-alias-border-l2);border-radius:11px;padding:13px 14px;color:var(--dsw-alias-label-tertiary);font-size:12px;line-height:1.6}
.reme-env-hint strong{color:var(--dsw-alias-label-primary);font-size:13px}
.reme-env-factrow{display:flex;align-items:center;gap:9px}
.reme-env-factrow .reme-env-path{flex:1}
.reme-env-field-label{flex:none;color:var(--dsw-alias-label-secondary);font-size:11.5px;line-height:1.5}
.reme-env-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.reme-env-metric{min-width:0;border:1px solid var(--dsw-alias-border-l2);border-radius:10px;background:color-mix(in srgb,var(--dsw-alias-bg-module-platform) 45%,var(--dsw-alias-bg-layer-3));padding:9px 11px}
.reme-env-metric>span{display:block;color:var(--dsw-alias-label-tertiary);font-size:10.5px;line-height:1.4}
.reme-env-metric>div{display:flex;min-width:0;flex-wrap:wrap;align-items:center;gap:5px;margin-top:6px}
.reme-env-metric small{color:var(--dsw-alias-label-tertiary);font-size:10px;line-height:1.4}
.reme-env-metric .reme-status-badge{max-width:100%;white-space:normal;align-items:flex-start;text-align:left}
.reme-env-metric .reme-status-badge i{margin-top:5px}
.reme-env-metric-path{min-width:0;overflow:hidden;color:var(--dsw-alias-label-primary);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:10.5px;line-height:1.5;text-overflow:ellipsis;white-space:nowrap}
.reme-env-cta{display:flex;flex-wrap:wrap;align-items:center;gap:8px;border-top:1px solid var(--dsw-alias-border-l2);padding-top:11px}
.reme-env-confirm{flex:1 0 100%;display:flex;flex-wrap:wrap;align-items:center;gap:8px;border:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 25%,var(--dsw-alias-border-l2));border-radius:9px;background:color-mix(in srgb,var(--dsw-alias-brand-primary) 5%,transparent);padding:9px 12px}
.reme-env-confirm p{flex:1 0 100%;margin:0;color:var(--dsw-alias-label-secondary);font-size:12px;line-height:1.5}
.reme-env-mirror{display:flex;align-items:center;gap:6px;color:var(--dsw-alias-label-secondary);font-size:12px}
.reme-env-mirror select{min-height:28px;border:1px solid var(--dsw-alias-border-l2);border-radius:7px;background:var(--dsw-alias-bg-layer-3);color:var(--dsw-alias-label-primary);padding:2px 8px;font:inherit;font-size:12px}
.reme-env-autostart{margin-left:auto;border:0;padding:0;color:var(--dsw-alias-label-secondary);font-size:12px}
.reme-env-progress{display:flex;flex-direction:column;gap:8px}
.reme-env-log{max-height:220px;overflow:auto;margin:0;border-radius:8px;background:var(--dsw-alias-bg-module-platform);padding:10px 12px;font-size:11px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.reme-env-result{display:flex;flex-direction:column;align-items:flex-start;gap:8px;border-top:1px solid var(--dsw-alias-border-l2);padding-top:11px}
.reme-env-result p{margin:0}
.reme-env-actions{display:flex;flex-wrap:wrap;align-items:center;gap:8px;border-top:1px solid var(--dsw-alias-border-l2);padding-top:11px}
.reme-env-inline{display:flex;align-items:center;gap:8px}
.reme-env-inline input{flex:1;min-width:0}
.reme-env-config{padding:2px 0 0}
.reme-env-group{margin:12px 0 0;color:var(--dsw-alias-label-secondary);font-size:11px;font-weight:650;letter-spacing:.02em}
.reme-env .reme-settings-details{margin:0}
.reme-env .reme-settings-details pre{margin:8px 0 0}
@media(max-width:820px){.reme-env-hero{grid-template-columns:1fr}.reme-env-hero-chips{justify-content:flex-start}.reme-env-metrics{grid-template-columns:1fr}.reme-env-option-path{display:none}.reme-env-autostart{margin-left:0}.reme-env-hero-meta{margin-left:0;width:100%}}
`;
