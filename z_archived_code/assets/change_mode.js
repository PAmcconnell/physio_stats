if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.clientside = {
    switchMode: function(trigger) {
        console.log("switchMode triggered", trigger);  // Debugging log
        if(trigger) {
            return {'mode': 'peak_correction'};
        }
        return window.dash_clientside.no_update;
    }
}