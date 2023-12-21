if (!window.dash_clientside) { window.dash_clientside = {}; }
window.dash_clientside.clientside = {
    setupGraphEventListener: function(trigger) {
        if(trigger === 'execute-js') {
            const graph = document.getElementById('ppg-plot');
            graph.on('plotly_selected', function(eventData) {
                if(eventData) {
                    const xRange = eventData.range.x;
                    const hiddenInput = document.getElementById('hidden-selection-range');
                    hiddenInput.value = JSON.stringify(xRange);
                    hiddenInput.dispatchEvent(new Event('change'));
                }
            });
        }
        return '';  // Return an empty string to reset the trigger
    }
}