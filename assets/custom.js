if (!window.dash_clientside) { window.dash_clientside = {}; }
window.dash_clientside.clientside = {
    setupGraphEventListener: function(trigger) {
        if(trigger === 'execute-js') {
            console.log("Executing JS to set up event listener");
            const graph = document.getElementById('ppg-plot');
            if (graph) {
                console.log("Graph element found");
                graph.on('plotly_selected', function(eventData) {
                    console.log("Box selection event triggered");
                    if(eventData) {
                        const xRange = eventData.range.x;
                        const hiddenInput = document.getElementById('hidden-selection-range');
                        if (hiddenInput) {
                            hiddenInput.value = JSON.stringify(xRange);
                            hiddenInput.dispatchEvent(new Event('change'));
                            console.log("Hidden input updated with range:", xRange);
                        } else {
                            console.error('Hidden input not found');
                        }
                    }
                });
            } else {
                console.error('Graph element not found');
            }
        }
        return '';  // Return an empty string to reset the trigger
    }
}
