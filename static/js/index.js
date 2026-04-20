window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Navbar burger toggle
    $(".navbar-burger").click(function() {
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");
    });
});

// ScenarioControl interactive galleries
(function () {
    // Each entry pairs a conditioning image with its rollout video.
    // Swap the `video` field to change which seed is shown for a given image.
    var imgCondEntries = [
        { image: "76da778ff251508d.jpg", video: "76da778ff251508d.mp4" },
        { image: "bbe92413054e59c4.jpg", video: "bbe92413054e59c4.mp4" },
        { image: "6a3775d893ef5c11.jpg", video: "6a3775d893ef5c11.mp4" },
        { image: "0bd948440a695247.jpg", video: "0bd948440a695247.mp4" },
        { image: "0fc75a8299825a12.jpg", video: "0fc75a8299825a12.mp4" },
        { image: "2fd9c9c8aa745ccb.jpg", video: "2fd9c9c8aa745ccb.mp4" },
        { image: "32db5c8feb07567c.jpg", video: "32db5c8feb07567c.mp4" },
        { image: "3d4fa046a5ad5f9d.jpg", video: "3d4fa046a5ad5f9d.mp4" },
        { image: "4fc9d3cec33b559b.jpg", video: "4fc9d3cec33b559b.mp4" },
    ];

    var selectedCondition = 1;

    var thumbsContainer = document.getElementById("img-cond-thumbs");
    var condImageEl = document.getElementById("selected-cond-image");
    var sceneVideoEl = document.getElementById("selected-scene-video");
    var sceneVideoSourceEl = sceneVideoEl ? sceneVideoEl.querySelector("source") : null;
    var scenePhaseLabelEl = document.getElementById("scene-phase-label");
    var sceneProgressBarEl = document.getElementById("scene-progress-bar");

    // How long to hold on frame 0 (the conditioning frame) before each rollout play.
    var SCENE_VIDEO_HOLD_MS = 2000;
    var sceneVideoHoldTimeoutId = null;
    var sceneProgressRafId = null;
    var sceneHoldStartTime = 0;
    var scenePhase = "hold"; // "hold" (showing frame 0) | "play" (rollout)

    var projConditionIndices = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14];
    var selectedProjCondition = 1;
    var projShowingGenerated = false;

    var projThumbsContainer = document.getElementById("img-cond-proj-thumbs");
    var projResultImageEl = document.getElementById("selected-proj-result-image");
    var projPhaseLabel = document.getElementById("proj-phase-label");

    var textCondConfigs = [
        { intersection: "four-way", curvature: "straight", pedestrians: "no",  lanes: "more", sources: [1, 7], prompt: "The scene is a four-way intersection featuring multiple lanes in each direction. Multiple vehicles exist." },
        { intersection: "T",        curvature: "straight", pedestrians: "no",  lanes: "less", sources: [2, 8], prompt: "The scene is a T-intersection. Multiple vehicles exist." },
        { intersection: "none",     curvature: "straight", pedestrians: "no",  lanes: "more", sources: [4],    prompt: "The scene depicts a multi-lane road with a left-turning ahead." },
        { intersection: "none",     curvature: "straight", pedestrians: "no",  lanes: "less", sources: [5],    prompt: "The scene depicts a narrow, vertically oriented road." },
        { intersection: "complex",  curvature: "straight", pedestrians: "no",  lanes: "more", sources: [6],    prompt: "The scene depicts a multi-lane intersection with a complex road topology." },
        { intersection: "none",     curvature: "curved",   pedestrians: "yes", lanes: "more", sources: [3, 9], prompt: "The scene depicts a multi-lane road with a curved layout. There are also pedestrians near the road." }
    ];
    var textCondCategories = ["intersection", "pedestrians", "lanes"];
    var textCondSelection = { intersection: "four-way", pedestrians: "no", lanes: "more" };
    var textCondCurrentConfig = null;
    var textCondPage = 0;

    var textCondPromptEl = document.getElementById("selected-text-cond-prompt");
    var textCondCarouselTrack = document.getElementById("text-cond-carousel-track");
    var textCondCarousel = document.getElementById("text-cond-carousel");
    var textCondPageLabel = document.getElementById("text-cond-page-label");
    var textCondPrevBtn = document.getElementById("text-cond-prev");
    var textCondNextBtn = document.getElementById("text-cond-next");

    function getCondImagePath(condIndex) {
        return "media/img_cond/images/" + imgCondEntries[condIndex - 1].image;
    }

    function getSceneVideoPath(condIndex) {
        return "media/img_cond/movies/" + imgCondEntries[condIndex - 1].video;
    }

    function getProjOrigPath(condIndex) {
        return "media/img_cond_proj/img_cond_proj_" + condIndex + "_orig.jpg";
    }

    function getProjMatchedPath(condIndex) {
        return "media/img_cond_proj/img_cond_proj_" + condIndex + "_orig_matched.jpg";
    }

    function getProjGeneratedPath(condIndex) {
        return "media/img_cond_proj/img_cond_proj_" + condIndex + ".jpg";
    }

    function renderThumbs() {
        while (thumbsContainer.firstChild) {
            thumbsContainer.removeChild(thumbsContainer.firstChild);
        }
        imgCondEntries.forEach(function (entry, idx) {
            var condIndex = idx + 1;
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "thumb-btn" + (condIndex === selectedCondition ? " active" : "");
            btn.setAttribute("aria-label", "Select conditioning image " + condIndex);

            var img = document.createElement("img");
            img.src = getCondImagePath(condIndex);
            img.alt = "Conditioning preview " + condIndex;

            btn.appendChild(img);
            btn.addEventListener("click", function () {
                selectedCondition = condIndex;
                updateView();
            });

            btn.addEventListener("mouseenter", function () {
                btn.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            });

            btn.addEventListener("focus", function () {
                btn.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            });

            thumbsContainer.appendChild(btn);
        });
    }

    function renderProjThumbs() {
        if (!projThumbsContainer) {
            return;
        }

        projThumbsContainer.innerHTML = "";
        projConditionIndices.forEach(function (condIndex) {
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "thumb-btn" + (condIndex === selectedProjCondition ? " active" : "");
            btn.setAttribute("aria-label", "Select projection conditioning image " + condIndex);

            var img = document.createElement("img");
            img.src = getProjOrigPath(condIndex);
            img.alt = "Projection conditioning preview " + condIndex;

            btn.appendChild(img);
            btn.addEventListener("click", function () {
                selectedProjCondition = condIndex;
                projShowingGenerated = false;
                updateProjView();
            });

            btn.addEventListener("mouseenter", function () {
                btn.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            });

            btn.addEventListener("focus", function () {
                btn.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            });

            projThumbsContainer.appendChild(btn);
        });
    }

    var interactiveViewEl = document.querySelector("#img-cond-interactive .interactive-view");

    function syncAspectRatios() {
        if (!interactiveViewEl) return;
        var condW = condImageEl.naturalWidth;
        var condH = condImageEl.naturalHeight;
        if (condW && condH) interactiveViewEl.style.setProperty("--cond-aspect", (condW / condH).toFixed(4));
        if (sceneVideoEl) {
            var vw = sceneVideoEl.videoWidth;
            var vh = sceneVideoEl.videoHeight;
            if (vw && vh) interactiveViewEl.style.setProperty("--scene-aspect", (vw / vh).toFixed(4));
        }
    }

    condImageEl.addEventListener("load", syncAspectRatios);
    if (sceneVideoEl) {
        sceneVideoEl.addEventListener("loadedmetadata", syncAspectRatios);
    }

    // Pause on frame 0 for a moment, then play the rollout. On end, repeat — this
    // gives the viewer a chance to anchor on the conditioning frame each loop.
    // The phase label and progress bar reflect which phase we're in.
    function setScenePhase(phase) {
        scenePhase = phase;
        if (scenePhaseLabelEl) {
            scenePhaseLabelEl.textContent = phase === "hold" ? "Conditional Generation" : "Rollout";
        }
        if (sceneProgressBarEl) {
            sceneProgressBarEl.style.width = "0%";
        }
    }

    function cancelSceneProgressTick() {
        if (sceneProgressRafId !== null) {
            cancelAnimationFrame(sceneProgressRafId);
            sceneProgressRafId = null;
        }
    }

    function tickSceneProgress() {
        sceneProgressRafId = requestAnimationFrame(tickSceneProgress);
        if (!sceneProgressBarEl || !sceneVideoEl) return;
        var progress;
        if (scenePhase === "hold") {
            progress = Math.min(1, (performance.now() - sceneHoldStartTime) / SCENE_VIDEO_HOLD_MS);
        } else {
            var dur = sceneVideoEl.duration;
            progress = (dur && isFinite(dur) && dur > 0)
                ? Math.min(1, sceneVideoEl.currentTime / dur)
                : 0;
        }
        sceneProgressBarEl.style.width = (progress * 100).toFixed(2) + "%";
    }

    function scheduleSceneVideoCycle() {
        if (!sceneVideoEl) return;
        clearTimeout(sceneVideoHoldTimeoutId);
        cancelSceneProgressTick();
        sceneVideoEl.pause();
        try { sceneVideoEl.currentTime = 0; } catch (e) { /* seek may fail if not seekable yet */ }
        setScenePhase("hold");
        sceneHoldStartTime = performance.now();
        tickSceneProgress();
        sceneVideoHoldTimeoutId = setTimeout(function () {
            setScenePhase("play");
            var p = sceneVideoEl.play();
            if (p && typeof p.catch === "function") p.catch(function () {});
        }, SCENE_VIDEO_HOLD_MS);
    }

    if (sceneVideoEl) {
        // Fires after the new source is ready (initial load and every source swap).
        sceneVideoEl.addEventListener("loadeddata", scheduleSceneVideoCycle);
        // Manual loop — so we can hold on frame 0 between plays.
        sceneVideoEl.addEventListener("ended", scheduleSceneVideoCycle);
    }

    function updateView() {
        condImageEl.src = getCondImagePath(selectedCondition);
        if (sceneVideoEl && sceneVideoSourceEl) {
            clearTimeout(sceneVideoHoldTimeoutId);
            cancelSceneProgressTick();
            sceneVideoSourceEl.src = getSceneVideoPath(selectedCondition);
            sceneVideoEl.load(); // triggers loadeddata → scheduleSceneVideoCycle
        }
        renderThumbs();
    }

    function updateProjView() {
        if (!projResultImageEl) {
            return;
        }

        projResultImageEl.src = projShowingGenerated
            ? getProjGeneratedPath(selectedProjCondition)
            : getProjMatchedPath(selectedProjCondition);

        if (projPhaseLabel) {
            projPhaseLabel.textContent = projShowingGenerated ? "Conditional Generation" : "Input \u2192 Projection";
        }

        renderProjThumbs();
    }

    function getTextCondSamples(config) {
        var samples = [];
        config.sources.forEach(function (srcId) {
            for (var s = 1; s <= 3; s++) {
                samples.push("media/text_cond/text_cond_" + srcId + "_" + s + ".jpg");
            }
        });
        return samples;
    }

    function findExactConfig(selection) {
        for (var i = 0; i < textCondConfigs.length; i++) {
            var c = textCondConfigs[i];
            var match = true;
            for (var j = 0; j < textCondCategories.length; j++) {
                if (c[textCondCategories[j]] !== selection[textCondCategories[j]]) {
                    match = false;
                    break;
                }
            }
            if (match) return c;
        }
        return null;
    }

    function findNearestConfig(selection, lockedCat) {
        var bestConfig = textCondConfigs[0];
        var bestDist = Infinity;
        for (var i = 0; i < textCondConfigs.length; i++) {
            if (lockedCat && textCondConfigs[i][lockedCat] !== selection[lockedCat]) continue;
            var dist = 0;
            for (var j = 0; j < textCondCategories.length; j++) {
                if (textCondConfigs[i][textCondCategories[j]] !== selection[textCondCategories[j]]) dist++;
            }
            if (dist < bestDist) {
                bestDist = dist;
                bestConfig = textCondConfigs[i];
            }
        }
        return bestConfig;
    }

    function computeAvailability(selection) {
        var availability = {};
        textCondCategories.forEach(function (targetCat) {
            availability[targetCat] = {};
            var seen = {};
            textCondConfigs.forEach(function (c) {
                var val = c[targetCat];
                if (seen[val]) return;
                if (selection[targetCat] === val) {
                    availability[targetCat][val] = true;
                    seen[val] = true;
                    return;
                }
                // Find nearest config that has this value (locked)
                var tempSel = { intersection: selection.intersection,
                    pedestrians: selection.pedestrians, lanes: selection.lanes };
                tempSel[targetCat] = val;
                var best = findNearestConfig(tempSel, targetCat);
                // Count how many OTHER categories would change
                var otherChanges = 0;
                textCondCategories.forEach(function (cat) {
                    if (cat !== targetCat && best[cat] !== selection[cat]) otherChanges++;
                });
                availability[targetCat][val] = otherChanges === 0;
                seen[val] = true;
            });
        });
        return availability;
    }

    function updateTextCondView() {
        if (!textCondCarouselTrack || !textCondPromptEl) return;
        var samples = getTextCondSamples(textCondCurrentConfig);
        var totalPages = Math.ceil(samples.length / 3);

        // Update carousel images
        var imgs = textCondCarouselTrack.querySelectorAll("img");
        for (var i = 0; i < 3; i++) {
            var idx = textCondPage * 3 + i;
            if (idx < samples.length) {
                imgs[i].src = samples[idx];
                imgs[i].style.display = "";
            } else {
                imgs[i].style.display = "none";
            }
        }

        // Update prompt
        textCondPromptEl.textContent = textCondCurrentConfig.prompt;

        // Update page label and chevron visibility
        if (totalPages > 1) {
            textCondCarousel.classList.add("has-pages");
            textCondPageLabel.textContent = "Page " + (textCondPage + 1) + " / " + totalPages;
            textCondPageLabel.style.display = "";
        } else {
            textCondCarousel.classList.remove("has-pages");
            textCondPageLabel.style.display = "none";
        }

        // Update pill states
        var availability = computeAvailability(textCondSelection);
        var pills = document.querySelectorAll("#text-cond-configurator .config-pill");
        for (var i = 0; i < pills.length; i++) {
            var pill = pills[i];
            var cat = pill.closest(".config-category").getAttribute("data-category");
            var val = pill.getAttribute("data-value");
            if (textCondSelection[cat] === val) {
                pill.classList.add("active");
            } else {
                pill.classList.remove("active");
            }
            if (availability[cat] && availability[cat][val]) {
                pill.classList.remove("unavailable");
            } else {
                pill.classList.add("unavailable");
            }
        }
    }

    function resolveTextCondConfig(lockedCat) {
        var config = findExactConfig(textCondSelection);
        if (!config) {
            config = findNearestConfig(textCondSelection, lockedCat);
            textCondCategories.forEach(function (cat) {
                textCondSelection[cat] = config[cat];
            });
        }
        textCondCurrentConfig = config;
        textCondPage = 0;
        updateTextCondView();
    }

    var configuratorPills = document.querySelectorAll("#text-cond-configurator .config-pill");
    for (var pi = 0; pi < configuratorPills.length; pi++) {
        (function (pill) {
            pill.addEventListener("click", function () {
                var cat = pill.closest(".config-category").getAttribute("data-category");
                var val = pill.getAttribute("data-value");
                textCondSelection[cat] = val;
                resolveTextCondConfig(cat);
            });
        })(configuratorPills[pi]);
    }

    if (textCondPrevBtn) {
        textCondPrevBtn.addEventListener("click", function () {
            var totalPages = Math.ceil(getTextCondSamples(textCondCurrentConfig).length / 3);
            textCondPage = (textCondPage - 1 + totalPages) % totalPages;
            updateTextCondView();
        });
    }
    if (textCondNextBtn) {
        textCondNextBtn.addEventListener("click", function () {
            var totalPages = Math.ceil(getTextCondSamples(textCondCurrentConfig).length / 3);
            textCondPage = (textCondPage + 1) % totalPages;
            updateTextCondView();
        });
    }

    var projImageBox = document.getElementById("proj-image-box");
    if (projImageBox) {
        projImageBox.addEventListener("mouseenter", function () {
            projShowingGenerated = true;
            updateProjView();
        });
        projImageBox.addEventListener("mouseleave", function () {
            projShowingGenerated = false;
            updateProjView();
        });
    }

    updateView();
    updateProjView();
    resolveTextCondConfig();
})();

// Additional prompt examples
(function () {
    var extraPrompts = [
        "ROAD & CONTROL: A two-lane road approaching a T-intersection with a right-turn lane from the right. AGENTS: The Ego vehicle is in the right-turn lane, preparing to turn right. Another is located on the right side, <mark>off the road near the intersection</mark>.",
        "ROAD & CONTROL: <mark>Two parallel straight lanes separated by a median</mark>; no intersections or traffic lights. AGENTS: Vehicles in adjacent lanes and approaching from opposite direction.",
        "ROAD & CONTROL: Road splits into <mark>left and right turn lanes</mark>; ego follows left-turn lane merging into a multi-lane road. AGENTS: <mark>A few vehicles ahead in the turn lane</mark>.",
        "ROAD & CONTROL: <mark>Five-lane road</mark> with a <mark>right-turn-only merge from the right</mark>. AGENTS: Several vehicles on the lanes.",
        "ROAD & CONTROL: Three-lane one-way oad with a <mark>left-turn-only lane intersecting from the left</mark>; <mark>all lanes green</mark>. AGENTS: Vehicles ahead in ego lane; <mark>pedestrians on right</mark>.",
        "ROAD & CONTROL: Two-lane road approaching a <mark>signalized intersection</mark> with east-west roads; <mark>dedicated turn and through lanes on all approaches</mark>. AGENTS: Several vehicles on the road.",
        "ROAD & CONTROL: The road has two lanes in a single direction, <mark>gently curving left ahead</mark>. No intersections, merges, splits, crosswalks, or traffic lights are visible along ego\u2019s path. AGENTS: The ego vehicle is centered in the left lane with <mark>no other agents or obstacles nearby</mark>."
    ];

    var extraSelected = 0;
    var extraTextEl = document.getElementById("extra-prompt-text");
    var extraImgEl = document.getElementById("extra-prompt-img");
    var extraPills = document.querySelectorAll(".extra-prompt-pill");

    if (!extraTextEl || !extraImgEl || !extraPills.length) return;

    function updateExtraPrompts() {
        extraTextEl.innerHTML = extraPrompts[extraSelected].replace(" AGENTS:", "<br><br>AGENTS:");
        extraImgEl.src = "media/text_cond_extra/" + (extraSelected + 1) + ".png";

        for (var i = 0; i < extraPills.length; i++) {
            if (parseInt(extraPills[i].getAttribute("data-extra-idx"), 10) === extraSelected) {
                extraPills[i].classList.add("active");
            } else {
                extraPills[i].classList.remove("active");
            }
        }
    }

    for (var i = 0; i < extraPills.length; i++) {
        (function (pill) {
            pill.addEventListener("click", function () {
                extraSelected = parseInt(pill.getAttribute("data-extra-idx"), 10);
                updateExtraPrompts();
            });
        })(extraPills[i]);
    }

    updateExtraPrompts();
})();

// Large Scene interactive map viewer
(function () {
    // focus: [x, y] in 0-1 range — point in the image to center on at max zoom
    var lsData = [
        { type: "image", cond: "media/large_scene/large_scene_1_cond.jpg", scene: "media/large_scene/large_scene_1_scene.jpg", focus: [0, 0.5] },
        { type: "image", cond: "media/large_scene/large_scene_2_cond.jpg", scene: "media/large_scene/large_scene_2_scene.jpg", focus: [0, 0.5] },
        { type: "text",  cond: "The scene depicts a multi-lane road intersection with a dedicated right-turn lane. The ego vehicle is positioned in the center, moving upward along a lane that merges into the right-turn path. To the left of the ego vehicle, there is a lane with multiple vehicles traveling in the same direction. To the right, there is a lane with multiple vehicles also traveling in the same direction, adjacent to the dedicated right-turn lane. There are pedestrians on the sidewalks and static objects such as road signs and barriers present.", scene: "media/large_scene/large_scene_3_scene.jpg", focus: [0, 1] },
        { type: "text",  cond: "The scene depicts a multi-lane intersection with a central roadway intersecting a perpendicular road from the left. The main road has two lanes in each direction. The perpendicular road has two lanes, one for each direction of travel. A traffic light at the intersection shows a green path allowing straight-through movement for the ego vehicle. Agents in the scene include multiple vehicles traveling in the same direction as the ego vehicle, others in the opposite direction or turning into the intersection, one pedestrian near the right-side crossing area, and one gray static object on the left side of the intersection.", scene: "media/large_scene/large_scene_4_scene.jpg", focus: [1, 1] }
    ];
    var lsCurrentTab = 0;

    var lsScale = 1, lsTransX = 0, lsTransY = 0;
    var lsMinScale = 1, lsMaxScale = 2;
    var lsNatWidth = 0, lsNatHeight = 0;
    var lsIsDragging = false;
    var lsDragStartX = 0, lsDragStartY = 0, lsDragStartTX = 0, lsDragStartTY = 0;
    var lsTouches = {};

    var lsTabBtns = document.querySelectorAll("[data-ls-tab]");
    var lsCondImageBox = document.getElementById("ls-cond-image-box");
    var lsCondImg = document.getElementById("ls-cond-img");
    var lsCondTextBox = document.getElementById("ls-cond-text-box");
    var lsCondText = document.getElementById("ls-cond-text");
    var lsMapContainer = document.getElementById("ls-map-container");
    var lsMapImg = document.getElementById("ls-map-img");

    if (!lsMapContainer) return;

    // Minimap
    var lsMinimap = document.createElement("div");
    lsMinimap.className = "large-scene-minimap";
    var lsMinimapImg = document.createElement("img");
    lsMinimapImg.draggable = false;
    lsMinimap.appendChild(lsMinimapImg);
    var lsMinimapRect = document.createElement("div");
    lsMinimapRect.className = "large-scene-minimap-rect";
    lsMinimap.appendChild(lsMinimapRect);
    lsMapContainer.appendChild(lsMinimap);

    function lsUpdateMinimap() {
        var cw = lsMapContainer.clientWidth;
        var ch = lsMapContainer.clientHeight;
        if (!lsNatWidth || !lsNatHeight || !cw || !ch) return;

        var mmW = lsMinimap.clientWidth;
        var mmH = lsMinimap.clientHeight;
        if (!mmW || !mmH) return;

        // Viewport rect in image-space (0-1)
        var vx = -lsTransX / (lsNatWidth * lsScale);
        var vy = -lsTransY / (lsNatHeight * lsScale);
        var vw = cw / (lsNatWidth * lsScale);
        var vh = ch / (lsNatHeight * lsScale);

        // Fit image into minimap (contain)
        var imgAspect = lsNatWidth / lsNatHeight;
        var mmAspect = mmW / mmH;
        var imgDrawW, imgDrawH, imgOffX, imgOffY;
        if (imgAspect > mmAspect) {
            imgDrawW = mmW;
            imgDrawH = mmW / imgAspect;
        } else {
            imgDrawH = mmH;
            imgDrawW = mmH * imgAspect;
        }
        imgOffX = (mmW - imgDrawW) / 2;
        imgOffY = (mmH - imgDrawH) / 2;

        // Position the minimap image
        lsMinimapImg.style.width = imgDrawW + "px";
        lsMinimapImg.style.height = imgDrawH + "px";
        lsMinimapImg.style.left = imgOffX + "px";
        lsMinimapImg.style.top = imgOffY + "px";

        // Position viewport rect
        lsMinimapRect.style.left = (imgOffX + vx * imgDrawW) + "px";
        lsMinimapRect.style.top = (imgOffY + vy * imgDrawH) + "px";
        lsMinimapRect.style.width = Math.min(vw * imgDrawW, imgDrawW) + "px";
        lsMinimapRect.style.height = Math.min(vh * imgDrawH, imgDrawH) + "px";

        // Hide rect when viewing entire image
        if (vw >= 0.99 && vh >= 0.99) {
            lsMinimapRect.style.display = "none";
        } else {
            lsMinimapRect.style.display = "";
        }
    }

    // Zoom hint (persistent, like Google Maps)
    var lsHint = document.createElement("div");
    lsHint.className = "large-scene-map-hint";
    var lsIsMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
    lsHint.textContent = (lsIsMac ? "\u2318" : "Ctrl") + " + scroll to zoom";
    lsMapContainer.appendChild(lsHint);

    // "Use cmd+scroll" warning tooltip
    var lsWarnHint = document.createElement("div");
    lsWarnHint.className = "large-scene-map-warn";
    lsWarnHint.textContent = "Use " + (lsIsMac ? "\u2318" : "Ctrl") + " + scroll to zoom";
    lsMapContainer.appendChild(lsWarnHint);
    var lsWarnTimer = null;

    function lsShowWarn() {
        lsWarnHint.classList.add("visible");
        clearTimeout(lsWarnTimer);
        lsWarnTimer = setTimeout(function () {
            lsWarnHint.classList.remove("visible");
        }, 1500);
    }

    function lsClamp() {
        var cw = lsMapContainer.clientWidth;
        var ch = lsMapContainer.clientHeight;
        var imgW = lsNatWidth * lsScale;
        var imgH = lsNatHeight * lsScale;

        if (imgW <= cw) {
            lsTransX = (cw - imgW) / 2;
        } else {
            if (lsTransX > 0) lsTransX = 0;
            if (lsTransX + imgW < cw) lsTransX = cw - imgW;
        }

        if (imgH <= ch) {
            lsTransY = (ch - imgH) / 2;
        } else {
            if (lsTransY > 0) lsTransY = 0;
            if (lsTransY + imgH < ch) lsTransY = ch - imgH;
        }
    }

    function lsApplyTransform() {
        lsClamp();
        lsMapImg.style.transform = "translate(" + lsTransX + "px, " + lsTransY + "px) scale(" + lsScale + ")";
        lsUpdateMinimap();
    }

    function lsFitImage() {
        var cw = lsMapContainer.clientWidth;
        var ch = lsMapContainer.clientHeight;
        if (!cw || !ch || !lsNatWidth || !lsNatHeight) return;

        lsMinScale = Math.min(cw / lsNatWidth, ch / lsNatHeight);
        lsMaxScale = Math.max(1, lsMinScale);

        // Start fully zoomed in, focused on the per-tab focus point
        lsScale = lsMaxScale;
        var focus = lsData[lsCurrentTab].focus;
        var imgW = lsNatWidth * lsScale;
        var imgH = lsNatHeight * lsScale;
        // Place the focus point at the center of the container
        lsTransX = cw / 2 - focus[0] * imgW;
        lsTransY = ch / 2 - focus[1] * imgH;
        lsApplyTransform();
    }

    function lsSyncHeight() {
        requestAnimationFrame(function () {
            // Match map height to just the conditioning content (image or text box), not the full column
            var condEl = lsCondImageBox.style.display !== "none" ? lsCondImageBox : lsCondTextBox;
            var contentH = condEl.offsetHeight;
            lsMapContainer.style.height = contentH + "px";
            lsFitImage();
        });
    }

    function lsSelectTab(index) {
        lsCurrentTab = index;
        var data = lsData[index];
        lsMinimapImg.src = data.scene;

        for (var i = 0; i < lsTabBtns.length; i++) {
            var li = lsTabBtns[i].closest("li");
            if (li) {
                li.classList.toggle("is-active", i === index);
            } else {
                lsTabBtns[i].classList.toggle("active", i === index);
            }
        }

        if (data.type === "image") {
            lsCondImageBox.style.display = "";
            lsCondTextBox.style.display = "none";
            lsCondImg.src = data.cond;
        } else {
            lsCondImageBox.style.display = "none";
            lsCondTextBox.style.display = "";
            lsCondText.textContent = data.cond;
        }

        lsMapImg.src = data.scene;
        lsMapImg.onload = function () {
            lsNatWidth = lsMapImg.naturalWidth;
            lsNatHeight = lsMapImg.naturalHeight;
            lsSyncHeight();
        };
        if (lsMapImg.complete && lsMapImg.naturalWidth > 0) {
            lsNatWidth = lsMapImg.naturalWidth;
            lsNatHeight = lsMapImg.naturalHeight;
            lsSyncHeight();
        }
    }

    // Tab click handlers
    for (var ti = 0; ti < lsTabBtns.length; ti++) {
        (function (btn, idx) {
            btn.addEventListener("click", function () { lsSelectTab(idx); });
        })(lsTabBtns[ti], ti);
    }

    // Re-sync height when conditioning image loads
    lsCondImg.addEventListener("load", function () { lsSyncHeight(); });

    // Re-sync on resize
    window.addEventListener("resize", function () { lsSyncHeight(); });

    // Zoom (cmd/ctrl + wheel)
    lsMapContainer.addEventListener("wheel", function (e) {
        if (!e.metaKey && !e.ctrlKey) {
            // Show warning tooltip, let page scroll normally
            lsShowWarn();
            return;
        }
        e.preventDefault();

        var rect = lsMapContainer.getBoundingClientRect();
        var cx = e.clientX - rect.left;
        var cy = e.clientY - rect.top;

        var zoomFactor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        var newScale = lsScale * zoomFactor;
        newScale = Math.max(lsMinScale, Math.min(lsMaxScale, newScale));

        var imgX = (cx - lsTransX) / lsScale;
        var imgY = (cy - lsTransY) / lsScale;
        lsScale = newScale;
        lsTransX = cx - imgX * lsScale;
        lsTransY = cy - imgY * lsScale;
        lsApplyTransform();
    }, { passive: false });

    // Pan (mouse)
    lsMapContainer.addEventListener("mousedown", function (e) {
        if (e.button !== 0) return;
        e.preventDefault();
        lsIsDragging = true;
        lsDragStartX = e.clientX;
        lsDragStartY = e.clientY;
        lsDragStartTX = lsTransX;
        lsDragStartTY = lsTransY;
        lsMapContainer.classList.add("is-grabbing");
    });

    document.addEventListener("mousemove", function (e) {
        if (!lsIsDragging) return;
        lsTransX = lsDragStartTX + (e.clientX - lsDragStartX);
        lsTransY = lsDragStartTY + (e.clientY - lsDragStartY);
        lsApplyTransform();
    });

    document.addEventListener("mouseup", function () {
        if (!lsIsDragging) return;
        lsIsDragging = false;
        lsMapContainer.classList.remove("is-grabbing");
    });

    // Touch support
    lsMapContainer.addEventListener("touchstart", function (e) {
        if (e.touches.length === 1) {
            var t = e.touches[0];
            lsIsDragging = true;
            lsDragStartX = t.clientX;
            lsDragStartY = t.clientY;
            lsDragStartTX = lsTransX;
            lsDragStartTY = lsTransY;
        } else if (e.touches.length === 2) {
            lsIsDragging = false;
            var t0 = e.touches[0], t1 = e.touches[1];
            lsTouches.dist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY);
            lsTouches.scale = lsScale;
            lsTouches.midX = (t0.clientX + t1.clientX) / 2;
            lsTouches.midY = (t0.clientY + t1.clientY) / 2;
            lsTouches.transX = lsTransX;
            lsTouches.transY = lsTransY;
        }
    }, { passive: true });

    lsMapContainer.addEventListener("touchmove", function (e) {
        e.preventDefault();
        if (e.touches.length === 1 && lsIsDragging) {
            var t = e.touches[0];
            lsTransX = lsDragStartTX + (t.clientX - lsDragStartX);
            lsTransY = lsDragStartTY + (t.clientY - lsDragStartY);
            lsApplyTransform();
        } else if (e.touches.length === 2 && lsTouches.dist) {
            var t0 = e.touches[0], t1 = e.touches[1];
            var newDist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY);
            var ratio = newDist / lsTouches.dist;
            var newScale = Math.max(lsMinScale, Math.min(lsMaxScale, lsTouches.scale * ratio));

            var rect = lsMapContainer.getBoundingClientRect();
            var mx = lsTouches.midX - rect.left;
            var my = lsTouches.midY - rect.top;
            var imgX = (mx - lsTouches.transX) / lsTouches.scale;
            var imgY = (my - lsTouches.transY) / lsTouches.scale;
            lsScale = newScale;
            lsTransX = mx - imgX * lsScale;
            lsTransY = my - imgY * lsScale;
            lsApplyTransform();
        }
    }, { passive: false });

    lsMapContainer.addEventListener("touchend", function () {
        lsIsDragging = false;
        lsTouches = {};
    }, { passive: true });

    // Initialize
    lsSelectTab(0);
})();

// Lazy-load videos: start loading when they scroll into view
(function () {
    var videos = document.querySelectorAll('video[preload="none"]');
    if (!('IntersectionObserver' in window)) {
        videos.forEach(function (v) { v.preload = 'auto'; v.load(); });
        return;
    }
    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                var video = entry.target;
                video.preload = 'auto';
                video.load();
                video.addEventListener('canplay', function () {
                    video.play().catch(function () {});
                }, { once: true });
                observer.unobserve(video);
            }
        });
    }, { rootMargin: '200px' });
    videos.forEach(function (v) { observer.observe(v); });
})();
